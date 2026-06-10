import { describe, it, expect, beforeEach } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, IfThenElseNode, ForKind
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
