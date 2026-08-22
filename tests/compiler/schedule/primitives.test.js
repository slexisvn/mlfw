import { describe, it, expect, beforeEach } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, IfThenElseNode, ForKind, SeqNode
} from '../../../src/compiler/ir/tensor/nodes.js';
import { PrimFunc } from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';

function iv(name) {
  return new VariableNode(name, 'int32');
}

function makeBlock(name, buf, indices, value) {
  const store = new BufferStoreNode(buf, indices, value);
  return new BlockNode(name, [], [], [{ buffer: buf }], store);
}

function loopNest(vars, extents, innerBody) {
  let body = innerBody;
  for (let i = vars.length - 1; i >= 0; i--) {
    body = new ForNode(vars[i], new IntImmNode(0), new IntImmNode(extents[i]), ForKind.SERIAL, body);
  }
  return body;
}

describe('Schedule primitives invalidate state cache', () => {
  beforeEach(() => resetVarCounter());

  it('split invalidates schedule state so loop bindings refresh', () => {
    const buf = new Buffer('A', [16], 'float32', 'global');
    const i = iv('i');
    const block = makeBlock('b', buf, [i], new IntImmNode(0));
    const body = loopNest([i], [16], block);
    const func = new PrimFunc('f', [], body);
    const sch = new Schedule(func);

    expect(sch.state.allLoopVarNames()).toEqual(['i']);
    const loops = sch.getLoops('b');
    sch.split(loops[0], 4);

    const names = sch.state.allLoopVarNames();
    expect(names).not.toContain('i');
    expect(names.length).toBe(2);
  });

  it('fuseLoops invalidates schedule state so loop bindings refresh', () => {
    const buf = new Buffer('A', [4, 4], 'float32', 'global');
    const i = iv('i');
    const j = iv('j');
    const block = makeBlock('b', buf, [i, j], new IntImmNode(0));
    const body = loopNest([i, j], [4, 4], block);
    const func = new PrimFunc('f', [], body);
    const sch = new Schedule(func);

    expect(sch.state.allLoopVarNames().sort()).toEqual(['i', 'j']);
    const loops = sch.getLoops('b');
    sch.fuseLoops(loops[0], loops[1]);

    const names = sch.state.allLoopVarNames();
    expect(names).not.toContain('i');
    expect(names).not.toContain('j');
    expect(names.length).toBe(1);
  });
});

describe('Schedule.reorder detaches loops from original parents', () => {
  beforeEach(() => resetVarCounter());

  it('produces an acyclic nest with no duplicated subtree', () => {
    const buf = new Buffer('A', [4, 8], 'float32', 'global');
    const i = iv('i');
    const j = iv('j');
    const block = makeBlock('b', buf, [i, j], new IntImmNode(0));
    const body = loopNest([i, j], [4, 8], block);
    const func = new PrimFunc('f', [], body);
    const sch = new Schedule(func);

    const loops = sch.getLoops('b');
    sch.reorder(loops[1], loops[0]);

    let node = func.body;
    const seen = new Set();
    const order = [];
    while (node && node.type === 'ForNode') {
      expect(seen.has(node)).toBe(false);
      seen.add(node);
      order.push(node.loopVar.name);
      node = node.body;
    }
    expect(node.type).toBe('BlockNode');
    expect(order).toEqual(['j', 'i']);
  });
});

describe('ScheduleValidator recurses into nested expressions', () => {
  it('flags rank-mismatched BufferLoad nested in a store value', () => {
    const dst = new Buffer('D', [8], 'float32', 'global');
    const srcBuf = new Buffer('S', [8, 8], 'float32', 'global');
    const i = iv('i');
    const badLoad = new BufferLoadNode(srcBuf, [i]);
    const block = makeBlock('b', dst, [i], badLoad);
    const body = loopNest([i], [8], block);
    const func = new PrimFunc('f', [], body);

    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => e.includes("'S'") && e.includes('rank mismatch'))).toBe(true);
  });

  it('flags rank-mismatched BufferLoad inside an IfThenElse body', () => {
    const dst = new Buffer('D', [8], 'float32', 'global');
    const srcBuf = new Buffer('S', [8, 8], 'float32', 'global');
    const i = iv('i');
    const badLoad = new BufferLoadNode(srcBuf, [i]);
    const innerStore = new BufferStoreNode(dst, [i], badLoad);
    const guard = new MathOpNode('<', i, new IntImmNode(4));
    const ite = new IfThenElseNode(guard, innerStore);
    const block = new BlockNode('b', [], [], [{ buffer: dst }], ite);
    const body = loopNest([i], [8], block);
    const func = new PrimFunc('f', [], body);

    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => e.includes("'S'") && e.includes('rank mismatch'))).toBe(true);
  });

  it('passes a well-formed nested access', () => {
    const dst = new Buffer('D', [8], 'float32', 'global');
    const srcBuf = new Buffer('S', [8], 'float32', 'global');
    const i = iv('i');
    const goodLoad = new BufferLoadNode(srcBuf, [i]);
    const block = makeBlock('b', dst, [i], goodLoad);
    const body = loopNest([i], [8], block);
    const func = new PrimFunc('f', [], body);

    expect(ScheduleValidator.validate(func)).toEqual([]);
  });
});

describe('ScheduleValidator flags ambiguous parallel partition (mismatched parallel extents)', () => {
  function parallelLoop(name, extent) {
    const buf = new Buffer(name, [extent], 'float32', 'global');
    const v = iv('p_' + name);
    const block = makeBlock('blk_' + name, buf, [v], new IntImmNode(0));
    return new ForNode(v, new IntImmNode(0), new IntImmNode(extent), ForKind.PARALLEL, block);
  }

  it('two sibling parallel loops with distinct extents is rejected', () => {
    const body = new SeqNode([parallelLoop('A', 4), parallelLoop('B', 5)]);
    const func = new PrimFunc('f', [], body);
    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => e.includes('parallel partition') && e.includes('extent 4') && e.includes('extent 5'))).toBe(true);
  });

  it('two sibling parallel loops with the same extent is allowed', () => {
    const body = new SeqNode([parallelLoop('A', 4), parallelLoop('B', 4)]);
    const func = new PrimFunc('f', [], body);
    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => e.includes('parallel partition'))).toBe(false);
  });

  it('a single parallel axis is allowed', () => {
    const body = new SeqNode([parallelLoop('A', 7)]);
    const func = new PrimFunc('f', [], body);
    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => e.includes('parallel partition'))).toBe(false);
  });
});

describe('Schedule.reorder decides on dependence, not on nest shape', () => {
  beforeEach(() => resetVarCounter());

  function opaqueNest(name, extents, writeOffsets, readOffsets) {
    const A = new Buffer(name, extents, 'float32', 'global');
    const vars = extents.map((_, d) => iv(`x${d}`));
    const at = (offsets) => offsets.map(([d, k]) => (k === 0 ? vars[d] : new MathOpNode('+', vars[d], new IntImmNode(k))));
    const store = new BufferStoreNode(A, at(writeOffsets), new BufferLoadNode(A, at(readOffsets)));
    return { func: new PrimFunc('f', [], loopNest(vars, extents, store), new Map([[name, A]])), vars };
  }

  function nestOrder(func) {
    const order = [];
    for (let node = func.body; node && node.type === 'ForNode'; node = node.body) order.push(node.loopVar.name);
    return order;
  }

  it('reorders two non-consecutive loops and leaves the loop between them in place', () => {
    const buf = new Buffer('A', [4, 5, 6], 'float32', 'global');
    const vars = [iv('i'), iv('j'), iv('k')];
    const block = makeBlock('b', buf, vars, new IntImmNode(0));
    const func = new PrimFunc('f', [], loopNest(vars, [4, 5, 6], block), new Map([['A', buf]]));
    const sch = new Schedule(func);

    const loops = sch.getLoops('b');
    sch.reorder(loops[2], loops[0]);

    expect(nestOrder(sch.func)).toEqual(['k', 'j', 'i']);
    expect(ScheduleValidator.validate(sch.func)).toEqual([]);
  });

  it('refuses an interchange that would reverse a skewed (<, >) dependence', () => {
    const { func, vars } = opaqueNest('A', [6, 6], [[0, 0], [1, 0]], [[0, -1], [1, 1]]);
    const sch = new Schedule(func);
    const loops = [sch._resolveLoop(vars[0].name), sch._resolveLoop(vars[1].name)];
    expect(() => sch.reorder(loops[1], loops[0])).toThrow(/violates a \w+ dependence on buffer 'A'/);
    expect(nestOrder(sch.func)).toEqual(['x0', 'x1']);
  });

  it('allows an interchange when every dependence stays lexicographically positive', () => {
    const { func, vars } = opaqueNest('A', [6, 6], [[0, 0], [1, 0]], [[0, -1], [1, -1]]);
    const sch = new Schedule(func);
    const loops = [sch._resolveLoop(vars[0].name), sch._resolveLoop(vars[1].name)];
    sch.reorder(loops[1], loops[0]);
    expect(nestOrder(sch.func)).toEqual(['x1', 'x0']);
  });

  it('sinks a split guard below the loop it constrains when that loop moves inward', () => {
    const buf = new Buffer('A', [12, 12], 'float32', 'global');
    const vars = [iv('i'), iv('j')];
    const block = makeBlock('b', buf, vars, new IntImmNode(0));
    const func = new PrimFunc('f', [], loopNest(vars, [12, 12], block), new Map([['A', buf]]));
    const sch = new Schedule(func);

    const loops = sch.getLoops('b');
    const [io, ii] = sch.split(loops[0], 8);
    const jLoop = sch.getLoops('b').find((l) => l.loopVar.name === 'j');
    sch.reorder(io, jLoop, ii);

    expect(ScheduleValidator.validate(sch.func)).toEqual([]);
    expect(nestOrder(sch.func)).toEqual([io.loopVar.name, 'j', ii.loopVar.name]);
  });
});

describe('split preserves a loop lower bound', () => {
  beforeEach(() => resetVarCounter());

  function shiftedNest(min, extent) {
    const A = new Buffer('A', [16], 'f32', 'global');
    const i = iv('i');
    const store = new BufferStoreNode(A, [i], new BufferLoadNode(A, [i]));
    const block = new BlockNode('b', [], [{ buffer: A }], [{ buffer: A }], store);
    const nest = new ForNode(i, new IntImmNode(min), new IntImmNode(extent), ForKind.SERIAL, block);
    return new PrimFunc('f', [], nest, new Map([['A', A]]));
  }

  function storedIndex(sch) {
    let found = null;
    const walk = (n) => {
      if (!n) return;
      if (n.type === 'BufferStoreNode') { found = n; return; }
      if (n.body) walk(n.body);
      if (n.stmts) n.stmts.forEach(walk);
    };
    walk(sch.func.body);
    return found;
  }

  it('offsets the split index by the loop min so the body visits the original range', () => {
    const func = shiftedNest(2, 4);
    const sch = new Schedule(func);

    sch.split(sch.getLoops('b')[0], 2);

    const store = storedIndex(sch);
    expect(store).not.toBeNull();
    const idx = store.indices[0];
    expect(idx.type).toBe('MathOpNode');
    expect(idx.op).toBe('+');
    expect(idx.a.type).toBe('IntImmNode');
    expect(idx.a.value).toBe(2);
  });

  it('emits no offset for the ordinary zero-based loop', () => {
    const func = shiftedNest(0, 4);
    const sch = new Schedule(func);

    sch.split(sch.getLoops('b')[0], 2);

    const idx = storedIndex(sch).indices[0];
    expect(idx.op).toBe('+');
    expect(idx.a.type).toBe('MathOpNode');
  });

  it('refuses to split a thread-bound loop, whose split has no single thread-axis meaning', () => {
    const func = shiftedNest(0, 8);
    const sch = new Schedule(func);
    const loop = sch.getLoops('b')[0];
    sch.bindThread(loop, 'threadIdx.x');

    expect(() => sch.split(sch.getLoops('b')[0], 2)).toThrow(/bound to/);
  });
});

describe('rfactor uses the reduction operator identity', () => {
  beforeEach(() => resetVarCounter());

  function reductionFunc(op) {
    const A = new Buffer('A', [4], 'f32', 'global');
    const C = new Buffer('C', [1], 'f32', 'global');
    const k = iv('k');
    const store = new BufferStoreNode(C, [new IntImmNode(0)],
      new MathOpNode(op, new BufferLoadNode(C, [new IntImmNode(0)]), new BufferLoadNode(A, [k])));
    const block = new BlockNode('r', [], [{ buffer: A }], [{ buffer: C }], store);
    const nest = new ForNode(k, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    return new PrimFunc('f', [], nest, new Map([['A', A], ['C', C]]));
  }

  function partialInits(sch) {
    const inits = [];
    const walk = (n) => {
      if (!n) return;
      if (n.type === 'BlockNode') {
        if (n.initBody && n.initBody.type === 'BufferStoreNode' && n.initBody.buffer.name.endsWith('_rf')) {
          inits.push(n.initBody.value);
        }
        walk(n.body);
        return;
      }
      if (n.body) walk(n.body);
      if (n.stmts) n.stmts.forEach(walk);
    };
    walk(sch.func.body);
    return inits;
  }

  it('initializes the partial buffer of a product reduction to 1, not 0', () => {
    const sch = new Schedule(reductionFunc('*'));

    sch.rfactor('r', 'k', 2);

    const inits = partialInits(sch);
    expect(inits.length).toBe(1);
    expect(inits[0].value).toBe(1);
  });

  it('initializes the partial buffer of a sum reduction to 0', () => {
    const sch = new Schedule(reductionFunc('+'));

    sch.rfactor('r', 'k', 2);

    expect(partialInits(sch)[0].value).toBe(0);
  });

  it('initializes a max reduction to negative infinity on a float accumulator', () => {
    const sch = new Schedule(reductionFunc('max'));

    sch.rfactor('r', 'k', 2);

    expect(partialInits(sch)[0].value).toBe(-Infinity);
  });

  it('refuses a store whose accumulator load is at a different subscript', () => {
    const A = new Buffer('A', [4], 'f32', 'global');
    const C = new Buffer('C', [2], 'f32', 'global');
    const k = iv('k');
    const store = new BufferStoreNode(C, [new IntImmNode(0)],
      new MathOpNode('+', new BufferLoadNode(C, [new IntImmNode(1)]), new BufferLoadNode(A, [k])));
    const block = new BlockNode('r', [], [{ buffer: A }], [{ buffer: C }], store);
    const nest = new ForNode(k, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const sch = new Schedule(new PrimFunc('f', [], nest, new Map([['A', A], ['C', C]])));

    expect(() => sch.rfactor('r', 'k', 2)).toThrow(/accumulation/);
  });

  it('refuses an update expression that itself reads the accumulator', () => {
    const A = new Buffer('A', [4], 'f32', 'global');
    const C = new Buffer('C', [1], 'f32', 'global');
    const k = iv('k');
    const zero = [new IntImmNode(0)];
    const update = new MathOpNode('*', new BufferLoadNode(A, [k]), new BufferLoadNode(C, zero));
    const store = new BufferStoreNode(C, zero, new MathOpNode('+', new BufferLoadNode(C, zero), update));
    const block = new BlockNode('r', [], [{ buffer: A }], [{ buffer: C }], store);
    const nest = new ForNode(k, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const sch = new Schedule(new PrimFunc('f', [], nest, new Map([['A', A], ['C', C]])));

    expect(() => sch.rfactor('r', 'k', 2)).toThrow(/reads accumulator/);
  });
});
