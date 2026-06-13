import { describe, it, expect, beforeEach } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, PrimFunc,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';
import { loopCarriesReduction } from '../../../src/compiler/schedule/legality.js';

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
    expect(() => sch.parallelize(loops.k)).toThrow(/reduction loop 'k'/);
    expect(loops.k.kind).toBe(ForKind.SERIAL);
    expect(() => sch.parallelize(loops.m)).not.toThrow();
    expect(loops.m.kind).toBe(ForKind.PARALLEL);
  });

  it('vectorize rejects the reduction loop k but allows the spatial loop n', () => {
    const sch = new Schedule(matmulFunc());
    const loops = loopByName(sch, 'mm');
    expect(() => sch.vectorize(loops.k)).toThrow(/vectorize reduction loop 'k'/);
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

  it('loopCarriesReduction names the offending block for k and returns null for m', () => {
    const sch = new Schedule(matmulFunc());
    const loops = loopByName(sch, 'mm');
    expect(loopCarriesReduction(loops.k)).toBe('mm');
    expect(loopCarriesReduction(loops.m)).toBe(null);
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
