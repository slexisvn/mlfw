import { describe, it, expect, beforeEach } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode, LetStmtNode, SeqNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, PrimFunc,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';
import { reductionLoopVars } from '../../../src/compiler/schedule/legality.js';
import { spatialIter, reduceIter } from '../../_utils/ir_fixture.js';

function matmulFunc(M = 8, K = 8, N = 8) {
  const A = new Buffer('A', [M, K], 'float32', 'global');
  const B = new Buffer('B', [K, N], 'float32', 'global');
  const C = new Buffer('C', [M, N], 'float32', 'global');
  const m = new VariableNode('m', 'int32');
  const n = new VariableNode('n', 'int32');
  const k = new VariableNode('k', 'int32');
  const prod = new MathOpNode('*', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n]));
  const acc = new MathOpNode('+', new BufferLoadNode(C, [m, n]), prod);
  const body = new BufferStoreNode(C, [m, n], acc);
  const init = new BufferStoreNode(C, [m, n], new IntImmNode(0));
  const iterVars = [{ iterVar: m, binding: m }, { iterVar: n, binding: n }, { iterVar: k, binding: k }];
  const block = new BlockNode('mm', iterVars, [{ buffer: A }, { buffer: B }], [{ buffer: C }], body, init);
  let nest = block;
  for (const [v, e] of [[k, K], [n, N], [m, M]]) {
    nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  }
  return new PrimFunc('mm', [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

function elementwiseFunc(M = 8, N = 8) {
  const A = new Buffer('A', [M, N], 'float32', 'global');
  const C = new Buffer('C', [M, N], 'float32', 'global');
  const m = new VariableNode('m', 'int32');
  const n = new VariableNode('n', 'int32');
  const body = new BufferStoreNode(C, [m, n], new MathOpNode('+', new BufferLoadNode(A, [m, n]), new IntImmNode(1)));
  const iterVars = [{ iterVar: m, binding: m }, { iterVar: n, binding: n }];
  const block = new BlockNode('ew', iterVars, [{ buffer: A }], [{ buffer: C }], body, null);
  let nest = block;
  for (const [v, e] of [[n, N], [m, M]]) {
    nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  }
  return new PrimFunc('ew', [], nest, new Map([['A', A], ['C', C]]));
}

const loopByName = (sch, blk) => Object.fromEntries(sch.getLoops(blk).map(l => [l.loopVar.name, l]));

describe('parallelize/vectorize reject reduction loops before mutating the IR', () => {
  beforeEach(() => resetVarCounter());

  it('parallelize rejects the reduction loop k but allows the spatial loops m, n', () => {
    const sch = new Schedule(matmulFunc());
    const loops = loopByName(sch, 'mm');
    expect(() => sch.parallelize(loops.k)).toThrow(/loop 'k' carries a \w+ dependence on buffer 'C'/);
    expect(loops.k.kind).toBe(ForKind.SERIAL);
    expect(() => sch.parallelize(loops.m)).not.toThrow();
    expect(loops.m.kind).toBe(ForKind.PARALLEL);
  });

  it('vectorize rejects the reduction loop k but allows the spatial loop n', () => {
    const sch = new Schedule(matmulFunc());
    const loops = loopByName(sch, 'mm');
    expect(() => sch.vectorize(loops.k)).toThrow(/Cannot vectorize: loop 'k' carries a \w+ dependence on buffer 'C'/);
    expect(loops.k.kind).toBe(ForKind.SERIAL);
    expect(() => sch.vectorize(loops.n)).not.toThrow();
    expect(loops.n.kind).toBe(ForKind.VECTORIZED);
  });

  it('every loop of a pure-elementwise block is parallel-safe', () => {
    const sch = new Schedule(elementwiseFunc());
    const loops = loopByName(sch, 'ew');
    expect(() => sch.parallelize(loops.m)).not.toThrow();
    expect(() => sch.vectorize(loops.n)).not.toThrow();
  });

  it('reductionLoopVars names k as the reduction axis of the matmul block and not m or n', () => {
    const sch = new Schedule(matmulFunc());
    expect([...reductionLoopVars(sch.getBlock('mm'))]).toEqual(['k']);
    expect([...reductionLoopVars(sch.getBlock('mm'))]).not.toContain('m');
  });

  it('keeps a declared reduction axis reducing when a second store indexes it', () => {
    // A fused reduction nest stores the elementwise value it is about to accumulate, so the
    // reduction axis does appear in a write index. Binding it to a thread would race the sum.
    const X = new Buffer('X', [4, 6], 'float32', 'global');
    const T = new Buffer('T', [4, 6], 'float32', 'global');
    const S = new Buffer('S', [4], 'float32', 'global');
    const i = new VariableNode('i', 'int32');
    const r = new VariableNode('r', 'int32');
    const t = new VariableNode('t', 'float32');
    const body = new SeqNode([
      new BufferStoreNode(T, [i, r], t),
      new BufferStoreNode(S, [i], new MathOpNode('+', new BufferLoadNode(S, [i]), t)),
    ]);
    const block = new BlockNode('fused', [spatialIter(i), reduceIter(r)],
      [{ buffer: X }], [{ buffer: S }, { buffer: T }],
      new LetStmtNode(t, new BufferLoadNode(X, [i, r]), body));
    let nest = block;
    for (const [v, e] of [[r, 6], [i, 4]]) {
      nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
    }
    const func = new PrimFunc('fused', [], nest, new Map([['X', X], ['T', T], ['S', S]]));

    const vars = reductionLoopVars(new Schedule(func).getBlock('fused'));
    expect([...vars]).toEqual(['r']);
    expect(vars.has('t')).toBe(false);
  });
});

describe('ScheduleValidator flags a parallel/vectorized reduction loop as a race', () => {
  it('flags a PARALLEL reduction loop k', () => {
    const func = matmulFunc();
    let cur = func.body;
    while (cur && cur.type === 'ForNode') { if (cur.loopVar.name === 'k') cur.kind = ForKind.PARALLEL; cur = cur.body; }
    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => /Parallel loop 'k' carries a reduction/.test(e))).toBe(true);
  });

  it('does not flag a PARALLEL spatial loop m', () => {
    const func = matmulFunc();
    let cur = func.body;
    while (cur && cur.type === 'ForNode') { if (cur.loopVar.name === 'm') cur.kind = ForKind.PARALLEL; cur = cur.body; }
    const errors = ScheduleValidator.validate(func);
    expect(errors.some(e => /carries a reduction/.test(e))).toBe(false);
  });
});

describe('loop-carried dependences the old name-matching reduction check could not see', () => {
  beforeEach(() => resetVarCounter());

  function recurrenceFunc(offset) {
    const A = new Buffer('A', [8], 'float32', 'global');
    const i = new VariableNode('i', 'int32');
    const read = offset === 0 ? i : new MathOpNode('+', i, new IntImmNode(offset));
    const store = new BufferStoreNode(A, [i], new MathOpNode('+', new BufferLoadNode(A, [read]), new IntImmNode(1)));
    const block = new BlockNode('r', [{ iterVar: i, binding: i }], [{ buffer: A }], [{ buffer: A }], store);
    const nest = new ForNode(i, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    return new PrimFunc('f', [], nest, new Map([['A', A]]));
  }

  it('refuses to parallelize A[i] = A[i-1] + 1, whose loop variable does appear in the write index', () => {
    const sch = new Schedule(recurrenceFunc(-1));
    expect(() => sch.parallelize(sch.getLoops('r')[0])).toThrow(/loop 'i' carries a \w+ dependence on buffer 'A'/);
    expect(sch.getLoops('r')[0].kind).toBe(ForKind.SERIAL);
  });

  it('refuses to parallelize the anti-dependence A[i] = A[i+1] + 1', () => {
    const sch = new Schedule(recurrenceFunc(1));
    expect(() => sch.parallelize(sch.getLoops('r')[0])).toThrow(/carries a \w+ dependence/);
  });

  it('allows parallelizing A[i] = A[i] + 1, where every iteration touches its own element', () => {
    const sch = new Schedule(recurrenceFunc(0));
    sch.parallelize(sch.getLoops('r')[0]);
    expect(sch.getLoops('r')[0].kind).toBe(ForKind.PARALLEL);
  });
});

describe('a loop whose extent is symbolic is checked for dependences like any other', () => {
  beforeEach(() => resetVarCounter());

  function matmulSymbolicK(M = 8, N = 8) {
    const A = new Buffer('A', [M, 8], 'float32', 'global');
    const B = new Buffer('B', [8, N], 'float32', 'global');
    const C = new Buffer('C', [M, N], 'float32', 'global');
    const m = new VariableNode('m', 'int32');
    const n = new VariableNode('n', 'int32');
    const k = new VariableNode('k', 'int32');
    const K = new VariableNode('K', 'int32');
    const prod = new MathOpNode('*', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n]));
    const acc = new MathOpNode('+', new BufferLoadNode(C, [m, n]), prod);
    const body = new BufferStoreNode(C, [m, n], acc);
    const init = new BufferStoreNode(C, [m, n], new IntImmNode(0));
    const iterVars = [{ iterVar: m, binding: m }, { iterVar: n, binding: n }, { iterVar: k, binding: k }];
    const block = new BlockNode('mm', iterVars, [{ buffer: A }, { buffer: B }], [{ buffer: C }], body, init);
    let nest = new ForNode(k, new IntImmNode(0), K, ForKind.SERIAL, block);
    nest = new ForNode(n, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, nest);
    nest = new ForNode(m, new IntImmNode(0), new IntImmNode(M), ForKind.SERIAL, nest);
    return new PrimFunc('mm', [K], nest, new Map([['A', A], ['B', B], ['C', C]]));
  }

  function rowwiseSymbolicBatch(offset) {
    const A = new Buffer('A', [8, 8], 'float32', 'global');
    const b = new VariableNode('b', 'int32');
    const j = new VariableNode('j', 'int32');
    const N = new VariableNode('N', 'int32');
    const read = offset === 0 ? j : new MathOpNode('+', j, new IntImmNode(offset));
    const store = new BufferStoreNode(A, [b, j], new MathOpNode('+', new BufferLoadNode(A, [b, read]), new IntImmNode(1)));
    const iterVars = [{ iterVar: b, binding: b }, { iterVar: j, binding: j }];
    const block = new BlockNode('ew', iterVars, [{ buffer: A }], [{ buffer: A }], store);
    let nest = new ForNode(j, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    nest = new ForNode(b, new IntImmNode(0), N, ForKind.SERIAL, nest);
    return new PrimFunc('ew', [N], nest, new Map([['A', A]]));
  }

  it('refuses to parallelize the reduction loop k when its extent is a symbol', () => {
    const sch = new Schedule(matmulSymbolicK());
    const loops = loopByName(sch, 'mm');
    expect(() => sch.parallelize(loops.k)).toThrow(/loop 'k' carries a \w+ dependence on buffer 'C'/);
    expect(loops.k.kind).toBe(ForKind.SERIAL);
  });

  it('still allows parallelizing the spatial loop m above a symbolic reduction loop', () => {
    const sch = new Schedule(matmulSymbolicK());
    const loops = loopByName(sch, 'mm');
    expect(() => sch.parallelize(loops.m)).not.toThrow();
    expect(loops.m.kind).toBe(ForKind.PARALLEL);
  });

  it('allows parallelizing a symbolic batch loop whose iterations touch disjoint rows', () => {
    const sch = new Schedule(rowwiseSymbolicBatch(0));
    const loops = loopByName(sch, 'ew');
    expect(() => sch.parallelize(loops.b)).not.toThrow();
    expect(loops.b.kind).toBe(ForKind.PARALLEL);
  });

  it('refuses to parallelize the recurrence inside a symbolic batch loop', () => {
    const sch = new Schedule(rowwiseSymbolicBatch(-1));
    const loops = loopByName(sch, 'ew');
    expect(() => sch.parallelize(loops.j)).toThrow(/loop 'j' carries a \w+ dependence on buffer 'A'/);
    expect(() => sch.parallelize(loops.b)).not.toThrow();
  });

  it('refuses an interchange that a symbolic outer level makes lexicographically negative', () => {
    const A = new Buffer('A', [8, 8], 'float32', 'global');
    const b = new VariableNode('b', 'int32');
    const j = new VariableNode('j', 'int32');
    const N = new VariableNode('N', 'int32');
    const read = new BufferLoadNode(A, [new MathOpNode('-', b, new IntImmNode(1)), new MathOpNode('+', j, new IntImmNode(1))]);
    const store = new BufferStoreNode(A, [b, j], new MathOpNode('+', read, new IntImmNode(1)));
    const block = new BlockNode('sk', [{ iterVar: b, binding: b }, { iterVar: j, binding: j }], [{ buffer: A }], [{ buffer: A }], store);
    let nest = new ForNode(j, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    nest = new ForNode(b, new IntImmNode(0), N, ForKind.SERIAL, nest);
    const sch = new Schedule(new PrimFunc('sk', [N], nest, new Map([['A', A]])));
    const loops = loopByName(sch, 'sk');

    expect(() => sch.reorder(loops.j, loops.b)).toThrow(/violates a \w+ dependence on buffer 'A'/);
  });
});

describe('iteration-variable kinds decide what a reduction axis may become', () => {
  beforeEach(() => resetVarCounter());

  function typedReduction() {
    const A = new Buffer('A', [8, 16], 'float32', 'global');
    const C = new Buffer('C', [8], 'float32', 'global');
    const s = new VariableNode('s', 'int32'), r = new VariableNode('r', 'int32');
    const vs = new VariableNode('vs', 'int32'), vr = new VariableNode('vr', 'int32');
    const store = new BufferStoreNode(C, [vs], new MathOpNode('+', new BufferLoadNode(C, [vs]), new BufferLoadNode(A, [vs, vr])));
    const block = new BlockNode('red', [spatialIter(vs, s), reduceIter(vr, r)],
      [{ buffer: A }], [{ buffer: C }], store, new BufferStoreNode(C, [vs], new IntImmNode(0)));
    let nest = new ForNode(r, new IntImmNode(0), new IntImmNode(16), ForKind.SERIAL, block);
    nest = new ForNode(s, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, nest);
    return new PrimFunc('f', [], nest, new Map([['A', A], ['C', C]]));
  }

  it('vectorizes a commutative-reduction axis (the backend lowers it to an accumulator)', () => {
    const sch = new Schedule(typedReduction());
    const loops = sch.getLoops('red');
    sch.vectorize(loops[1]);
    expect(loops[1].kind).toBe(ForKind.VECTORIZED);
  });

  it('still refuses to parallelize that same commutative-reduction axis', () => {
    const sch = new Schedule(typedReduction());
    const loops = sch.getLoops('red');
    expect(() => sch.parallelize(loops[1])).toThrow(/carries a \w+ dependence on buffer 'C'/);
    expect(loops[1].kind).toBe(ForKind.SERIAL);
  });

  it('parallelizes the spatial axis of the same block', () => {
    const sch = new Schedule(typedReduction());
    const loops = sch.getLoops('red');
    sch.parallelize(loops[0]);
    expect(loops[0].kind).toBe(ForKind.PARALLEL);
  });
});
