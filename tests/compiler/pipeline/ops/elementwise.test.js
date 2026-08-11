import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { countLoops, countTempBuffers } from '../../../_utils/kernel_source.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

function runKernel(result, name, inputs, outputShapes) {
  const inArrays = inputs.map(i => new Float32Array(i));
  const outArrays = outputShapes.map(s => {
    let n = 1;
    for (const d of s) n *= d;
    return new Float32Array(n);
  });
  result.run(name, ...inArrays, ...outArrays);
  return outArrays;
}

describe('elementwise single ops', () => {
  it('add two vectors element-wise', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const [out] = runKernel(result, 'add', [[1, 2, 3, 4], [10, 20, 30, 40]], [[4]]);
    expect(Array.from(out)).toEqual([11, 22, 33, 44]);
  });

  it('mul two matrices element-wise', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const [out] = runKernel(result, 'mul', [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], [[2, 3]]);
    expect(Array.from(out)).toEqual([2, 6, 12, 20, 30, 42]);
  });

  it('neg vector', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const [out] = runKernel(result, 'neg', [[1, -2, 3]], [[3]]);
    expect(Array.from(out)).toEqual([-1, 2, -3]);
  });

  it('exp vector', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('exp_f', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const [out] = runKernel(result, 'exp_f', [[0, 1, -1]], [[3]]);
    expect(out[0]).toBeCloseTo(1.0, 5);
    expect(out[1]).toBeCloseTo(Math.E, 5);
    expect(out[2]).toBeCloseTo(1 / Math.E, 5);
  });
});

describe('elementwise chains — passes optimize before codegen', () => {
  it('neg(neg(x)) optimized away by algebraic simplification', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('double_neg', [t], [t], (b, args) => {
      const n1 = b.neg(args[0]);
      const n2 = b.neg(n1.getResult(0));
      b.returnOp([n2.getResult(0)]);
    });

    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = result.getSource('double_neg');
    expect(src).not.toMatch(/0\s*-/);
    expect(countLoops(src)).toBe(1);
    const [out] = runKernel(result, 'double_neg', [[1, -2, 3, -4]], [[4]]);
    expect(Array.from(out)).toEqual([1, -2, 3, -4]);
  });

  it('add then mul chain — fused into single loop', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('add_mul', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('add_mul');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
    const [out] = runKernel(result, 'add_mul',
      [[1, 2, 3, 4], [10, 20, 30, 40], [2, 2, 2, 2]], [[4]]);
    expect(Array.from(out)).toEqual([22, 44, 66, 88]);
  });

  it('sub then neg chain — fused into single loop', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('sub_neg', [t, t], [t], (b, args) => {
      const diff = b.sub(args[0], args[1]);
      b.returnOp([b.neg(diff.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('sub_neg');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
    const [out] = runKernel(result, 'sub_neg', [[10, 20, 30], [1, 2, 3]], [[3]]);
    expect(Array.from(out)).toEqual([-9, -18, -27]);
  });

  it('multiple outputs — add and mul from same inputs', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('dual_out', [t, t], [t, t], (b, args) => {
      const s = b.add(args[0], args[1]);
      const p = b.mul(args[0], args[1]);
      b.returnOp([s.getResult(0), p.getResult(0)]);
    });

    const result = compile(func);
    const inA = new Float32Array([1, 2, 3]);
    const inB = new Float32Array([4, 5, 6]);
    const outSum = new Float32Array(3);
    const outProd = new Float32Array(3);
    result.run('dual_out', inA, inB, outSum, outProd);
    expect(Array.from(outSum)).toEqual([5, 7, 9]);
    expect(Array.from(outProd)).toEqual([4, 10, 18]);
  });
});

describe('elementwise with dead code — DCE removes unused ops', () => {
  it('dead neg branch is eliminated, only live path executes', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('dce_test', [t, t], [t], (b, args) => {
      b.neg(args[0]);
      b.exp(args[1]);
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = result.getSource('dce_test');
    expect(src).not.toMatch(/Math\.exp/);
    expect(src).not.toMatch(/0\s*-/);
    expect(countLoops(src)).toBe(1);
    const [out] = runKernel(result, 'dce_test', [[1, 2, 3, 4], [10, 20, 30, 40]], [[4]]);
    expect(Array.from(out)).toEqual([11, 22, 33, 44]);
  });
});

describe('broadcast materialization — explicit broadcast_dimensions in fusion', () => {
  it('add with broadcast [3] -> [4, 3] dims=[1] produces correct values', () => {
    const mat = new TensorType([4, 3], ScalarType.F32);
    const vec = new TensorType([3], ScalarType.F32);
    const func = buildFunction('bc_add', [mat, vec], [mat], (b, args) => {
      const bcast = b.broadcast(args[1], [4, 3], [1]);
      b.returnOp([b.add(args[0], bcast.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const [out] = runKernel(result, 'bc_add',
      [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [100, 200, 300]], [[4, 3]]);
    expect(Array.from(out)).toEqual([101, 202, 303, 104, 205, 306, 107, 208, 309, 110, 211, 312]);
  });

  it('add with broadcast [4] -> [4, 3] dims=[0] maps to first dim correctly', () => {
    const mat = new TensorType([4, 3], ScalarType.F32);
    const vec = new TensorType([4], ScalarType.F32);
    const func = buildFunction('bc_dim0', [mat, vec], [mat], (b, args) => {
      const bcast = b.broadcast(args[1], [4, 3], [0]);
      b.returnOp([b.add(args[0], bcast.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const [out] = runKernel(result, 'bc_dim0',
      [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [10, 20, 30, 40]], [[4, 3]]);
    expect(Array.from(out)).toEqual([11, 12, 13, 24, 25, 26, 37, 38, 39, 50, 51, 52]);
  });

  it('mul-then-add chain with broadcast on non-trailing dim', () => {
    const mat = new TensorType([2, 3], ScalarType.F32);
    const row = new TensorType([2], ScalarType.F32);
    const func = buildFunction('bc_chain', [mat, row], [mat], (b, args) => {
      const bcast = b.broadcast(args[1], [2, 3], [0]);
      const prod = b.mul(args[0], bcast.getResult(0));
      b.returnOp([b.neg(prod.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const [out] = runKernel(result, 'bc_chain',
      [[1, 2, 3, 4, 5, 6], [10, 20]], [[2, 3]]);
    expect(Array.from(out)).toEqual([-10, -20, -30, -80, -100, -120]);
  });
});

describe('codegen quality — constant buffer inlining in fusion', () => {
  it('broadcast scalar constant fused with elementwise produces no temp buffer', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('const_fuse', [t], [t], (b, args) => {
      const c = b.scalarConstant(2, ScalarType.F32);
      const bc = b.broadcast(c.getResult(0), [8], []);
      b.returnOp([b.mul(args[0], bc.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('const_fuse');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
    const [out] = runKernel(result, 'const_fuse', [[1, 2, 3, 4, 5, 6, 7, 8]], [[8]]);
    expect(Array.from(out)).toEqual([2, 4, 6, 8, 10, 12, 14, 16]);
  });

  it('multiple constant broadcasts in chain all inlined', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('multi_const', [t], [t], (b, args) => {
      const c1 = b.broadcast(b.scalarConstant(3, ScalarType.F32).getResult(0), [4], []);
      const c2 = b.broadcast(b.scalarConstant(10, ScalarType.F32).getResult(0), [4], []);
      const scaled = b.mul(args[0], c1.getResult(0)).getResult(0);
      b.returnOp([b.add(scaled, c2.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('multi_const');
    expect(countTempBuffers(src)).toBe(0);
    const [out] = runKernel(result, 'multi_const', [[1, 2, 3, 4]], [[4]]);
    expect(Array.from(out)).toEqual([13, 16, 19, 22]);
  });
});

describe('elementwise with CSE — common subexpressions eliminated', () => {
  it('duplicate add(a,b) computed once', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('cse_test', [t, t], [t], (b, args) => {
      const s1 = b.add(args[0], args[1]);
      const s2 = b.add(args[0], args[1]);
      b.returnOp([b.mul(s1.getResult(0), s2.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('cse_test');
    expect(src).toMatch(/\bcse\d+/);
    const addCount = (src.match(/buf_\d+\[i\d+_\d+\]\s*\+\s*buf_\d+\[i\d+_\d+\]/g) || []).length;
    expect(addCount).toBe(1);
    const [out] = runKernel(result, 'cse_test', [[1, 2, 3], [4, 5, 6]], [[3]]);
    expect(Array.from(out)).toEqual([25, 49, 81]);
  });
});
