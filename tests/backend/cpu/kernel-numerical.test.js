import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { countLoops as countForLoops } from '../../_utils/kernel_source.js';
import { compileScheduled as compile } from '../../_utils/ir_fixture.js';
import { compileUnscheduled } from '../../_utils/ir_fixture.js';

function src(result, name) {
  return result.getSource(name);
}


function countStores(s) {
  return (s.match(/\w+\[.*?\]\s*=/g) || []).length;
}

function hasNoopStore(s) {
  const lines = s.split('\n').map(l => l.trim());
  for (const line of lines) {
    const m = line.match(/^(\w+\[[^\]]+\])\s*=\s*(.+);$/);
    if (m && m[1].replace(/\s+/g, '') === m[2].replace(/\s+/g, '')) return true;
  }
  return false;
}

function extractUsedVars(s) {
  const declared = new Set();
  for (const m of s.matchAll(/\b(?:let|const|var)\s+(\w+)/g)) declared.add(m[1]);
  for (const m of s.matchAll(/function\s+\w+\(([^)]*)\)/g)) {
    for (const p of m[1].split(',')) {
      const name = p.trim();
      if (name) declared.add(name);
    }
  }
  declared.add('Math');
  declared.add('Float32Array');
  declared.add('Float64Array');
  declared.add('Int32Array');
  declared.add('Infinity');
  declared.add('undefined');
  return declared;
}


describe('CPU kernel quality — numerical correctness (run)', () => {
  it('add: [1,2,3] + [4,5,6] = [5,7,9]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3]);
    const b2 = new Float32Array([4, 5, 6]);
    const out = new Float32Array(3);
    result.run('c_run_add', a, b2, out);
    expect(Array.from(out)).toEqual([5, 7, 9]);
  });

  it('neg: -[1, -2, 3] = [-1, 2, -3]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, -2, 3]);
    const out = new Float32Array(3);
    result.run('c_run_neg', a, out);
    expect(Array.from(out)).toEqual([-1, 2, -3]);
  });

  it('mul: [2,3,4] * [5,6,7] = [10,18,28]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([2, 3, 4]);
    const b2 = new Float32Array([5, 6, 7]);
    const out = new Float32Array(3);
    result.run('c_run_mul', a, b2, out);
    expect(Array.from(out)).toEqual([10, 18, 28]);
  });

  it('sub: [10,20,30] - [1,2,3] = [9,18,27]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_sub', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([10, 20, 30]);
    const b2 = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    result.run('c_run_sub', a, b2, out);
    expect(Array.from(out)).toEqual([9, 18, 27]);
  });

  it('exp: exp([0, 1]) ≈ [1.0, 2.718]', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_run_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, 1]);
    const out = new Float32Array(2);
    result.run('c_run_exp', a, out);
    expect(out[0]).toBeCloseTo(1.0, 5);
    expect(out[1]).toBeCloseTo(Math.E, 4);
  });

  it('sqrt: sqrt([4, 9, 16]) = [2, 3, 4]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_sqrt', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([4, 9, 16]);
    const out = new Float32Array(3);
    result.run('c_run_sqrt', a, out);
    expect(out[0]).toBeCloseTo(2, 5);
    expect(out[1]).toBeCloseTo(3, 5);
    expect(out[2]).toBeCloseTo(4, 5);
  });

  it('add+mul fusion: (a+b)*c produces correct values', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('c_run_chain', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4]);
    const b2 = new Float32Array([10, 20, 30, 40]);
    const c = new Float32Array([2, 3, 4, 5]);
    const out = new Float32Array(4);
    result.run('c_run_chain', a, b2, c, out);
    expect(Array.from(out)).toEqual([22, 66, 132, 220]);
  });
});


describe('CPU kernel quality — reduction numerical correctness', () => {
  it('sum reduce [2,3]: row sums', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_run_rsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('c_run_rsum', a, out);
    expect(out[0]).toBeCloseTo(6, 5);
    expect(out[1]).toBeCloseTo(15, 5);
  });

  it('max reduce [2,4]: row maxes', () => {
    const tin = new TensorType([2, 4], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_run_rmax', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]);
    const out = new Float32Array(2);
    result.run('c_run_rmax', a, out);
    expect(out[0]).toBe(4);
    expect(out[1]).toBe(9);
  });

  it('sum reduce axis 0 [3,2]: column sums', () => {
    const tin = new TensorType([3, 2], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_run_csum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('c_run_csum', a, out);
    expect(out[0]).toBeCloseTo(9, 5);
    expect(out[1]).toBeCloseTo(12, 5);
  });
});


describe('CPU kernel quality — matmul numerical correctness', () => {
  it('identity matmul: I @ v = v', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 1], ScalarType.F32);
    const out = new TensorType([2, 1], ScalarType.F32);
    const func = buildFunction('c_run_mm_id', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const eye = new Float32Array([1, 0, 0, 1]);
    const v = new Float32Array([3, 7]);
    const o = new Float32Array(2);
    result.run('c_run_mm_id', eye, v, o);
    expect(o[0]).toBeCloseTo(3, 5);
    expect(o[1]).toBeCloseTo(7, 5);
  });

  it('2x3 @ 3x2: correct product', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_mm_23', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b2 = new Float32Array([7, 8, 9, 10, 11, 12]);
    const o = new Float32Array(4);
    result.run('c_run_mm_23', a, b2, o);
    expect(o[0]).toBeCloseTo(58, 3);
    expect(o[1]).toBeCloseTo(64, 3);
    expect(o[2]).toBeCloseTo(139, 3);
    expect(o[3]).toBeCloseTo(154, 3);
  });

  it('matmul + neg: -(A @ B) correct values', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_mmneg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([5, 3, 2, 7]);
    const o = new Float32Array(4);
    result.run('c_run_mmneg', a, b2, o);
    expect(o[0]).toBeCloseTo(-5, 5);
    expect(o[1]).toBeCloseTo(-3, 5);
    expect(o[2]).toBeCloseTo(-2, 5);
    expect(o[3]).toBeCloseTo(-7, 5);
  });

  it('matmul + bias + relu: max(A@B + bias, 0)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const bias = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_biasrelu', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([3, -5, 2, 4]);
    const bi = new Float32Array([-10, 1, 1, -10]);
    const o = new Float32Array(4);
    result.run('c_run_biasrelu', a, b2, bi, o);
    expect(o[0]).toBeCloseTo(0, 5);
    expect(o[1]).toBeCloseTo(0, 5);
    expect(o[2]).toBeCloseTo(3, 5);
    expect(o[3]).toBeCloseTo(0, 5);
  });
});


describe('CPU kernel quality — 2D elementwise numerical correctness', () => {
  it('[2,3] add: element-wise correct', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('c_run_2d_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b2 = new Float32Array([10, 20, 30, 40, 50, 60]);
    const out = new Float32Array(6);
    result.run('c_run_2d_add', a, b2, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 44, 55, 66]);
  });

  it('[2,3] neg: element-wise correct', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('c_run_2d_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, -2, 3, -4, 5, -6]);
    const out = new Float32Array(6);
    result.run('c_run_2d_neg', a, out);
    expect(Array.from(out)).toEqual([-1, 2, -3, 4, -5, 6]);
  });
});


describe('CPU kernel quality — edge cases', () => {
  it('1-element tensor compiles and runs', () => {
    const t = new TensorType([1], ScalarType.F32);
    const func = buildFunction('c_edge1', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const a = new Float32Array([5]);
    const b2 = new Float32Array([3]);
    const out = new Float32Array(1);
    result.run('c_edge1', a, b2, out);
    expect(out[0]).toBe(8);
  });

  it('non-power-of-2 sizes compile and produce correct results', () => {
    const t = new TensorType([7], ScalarType.F32);
    const func = buildFunction('c_edge7', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    const b2 = new Float32Array([10, 20, 30, 40, 50, 60, 70]);
    const out = new Float32Array(7);
    result.run('c_edge7', a, b2, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 44, 55, 66, 77]);
  });

  it('large tensor 10000: compiles and runs', () => {
    const t = new TensorType([10000], ScalarType.F32);
    const func = buildFunction('c_edge_big', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const a = new Float32Array(10000).fill(1);
    const b2 = new Float32Array(10000).fill(2);
    const out = new Float32Array(10000);
    result.run('c_edge_big', a, b2, out);
    expect(out[0]).toBe(3);
    expect(out[9999]).toBe(3);
  });
});


describe('CPU kernel quality — matmul size variants', () => {
  const configs = [
    { M: 1, K: 4, N: 1 },
    { M: 4, K: 4, N: 4 },
    { M: 3, K: 5, N: 7 },
  ];

  for (const { M, K, N } of configs) {
    it(`${M}x${K} @ ${K}x${N}: correct result`, () => {
      const lhs = new TensorType([M, K], ScalarType.F32);
      const rhs = new TensorType([K, N], ScalarType.F32);
      const out = new TensorType([M, N], ScalarType.F32);
      const func = buildFunction(`c_mm_${M}${K}${N}`, [lhs, rhs], [out], (b, args) => {
        b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
      });
      const result = compile(func);
      const a = new Float32Array(M * K).fill(1);
      const b2 = new Float32Array(K * N).fill(1);
      const o = new Float32Array(M * N);
      result.run(`c_mm_${M}${K}${N}`, a, b2, o);
      for (let i = 0; i < M * N; i++) {
        expect(o[i]).toBeCloseTo(K, 3);
      }
    });
  }
});


describe('CPU kernel quality — reduce + elementwise chain', () => {
  it('sum + neg: correct negated sum', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_run_sumneg', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const s = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      b.returnOp([b.neg(s.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('c_run_sumneg', a, out);
    expect(out[0]).toBeCloseTo(-6, 5);
    expect(out[1]).toBeCloseTo(-15, 5);
  });
});


describe('CPU kernel quality — dtype handling', () => {
  it('f64 add: uses Float64Array temp, correct values', () => {
    const t = new TensorType([4], ScalarType.F64);
    const func = buildFunction('c_f64_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const s = src(result, 'c_f64_add');
    expect(s).toMatch(/^function\s+c_f64_add\(/m);
    const a = new Float64Array([1.1, 2.2, 3.3, 4.4]);
    const b2 = new Float64Array([0.1, 0.2, 0.3, 0.4]);
    const out = new Float64Array(4);
    result.run('c_f64_add', a, b2, out);
    expect(out[0]).toBeCloseTo(1.2, 10);
    expect(out[3]).toBeCloseTo(4.8, 10);
  });

  it('f32 vs f64: same op, different typed arrays', () => {
    const t32 = new TensorType([4], ScalarType.F32);
    const t64 = new TensorType([4], ScalarType.F64);
    const f32 = buildFunction('c_dtype32', [t32, t32], [t32], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const f64 = buildFunction('c_dtype64', [t64, t64], [t64], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s32 = src(compile(f32), 'c_dtype32');
    const s64 = src(compile(f64), 'c_dtype64');
    expect(s32).not.toEqual(s64);
  });
});


describe('CPU kernel quality — long fusion chains', () => {
  it('5-op chain (add+mul+neg+exp+sqrt): single kernel, single store per element', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_chain5', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      const neg = b.neg(prod.getResult(0));
      const ex = b.exp(neg.getResult(0));
      b.returnOp([b.sqrt(ex.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const s = src(result, 'c_chain5');
    expect(s).toMatch(/Math\.sqrt/);
    expect(s).toMatch(/Math\.exp/);
  });

  it('5-op chain: numerically correct', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_chain5_run', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      const neg = b.neg(prod.getResult(0));
      const ex = b.exp(neg.getResult(0));
      b.returnOp([b.sqrt(ex.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2]);
    const b2 = new Float32Array([1, 1]);
    const c = new Float32Array([1, 1]);
    const out = new Float32Array(2);
    result.run('c_chain5_run', a, b2, c, out);
    expect(out[0]).toBeCloseTo(Math.sqrt(Math.exp(-(1 + 1) * 1)), 4);
    expect(out[1]).toBeCloseTo(Math.sqrt(Math.exp(-(2 + 1) * 1)), 4);
  });

  it('4-op chain: no intermediate temp buffers', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('c_chain4_notemp', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const neg = b.neg(sum.getResult(0));
      const ex = b.exp(neg.getResult(0));
      b.returnOp([b.abs(ex.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_chain4_notemp');
    expect(s).not.toMatch(/new Float32Array/);
    expect(s).not.toMatch(/new Float64Array/);
  });
});


describe('CPU kernel quality — full reduction (scalar output)', () => {
  it('sum all [16]: produces single scalar', () => {
    const tin = new TensorType([16], ScalarType.F32);
    const tout = new TensorType([], ScalarType.F32);
    const func = buildFunction('c_fullsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(16).fill(2);
    const out = new Float32Array(1);
    result.run('c_fullsum', a, out);
    expect(out[0]).toBeCloseTo(32, 5);
  });

  it('max all [8]: produces single max', () => {
    const tin = new TensorType([8], ScalarType.F32);
    const tout = new TensorType([], ScalarType.F32);
    const func = buildFunction('c_fullmax', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [0], 'max').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]);
    const out = new Float32Array(1);
    result.run('c_fullmax', a, out);
    expect(out[0]).toBe(9);
  });
});


describe('CPU kernel quality — relu', () => {
  it('relu: uses Math.max(0, x) pattern', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_relu_src', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const s = src(compile(func), 'c_relu_src');
    expect(s).toMatch(/Math\.max\(/);
    expect(s).toMatch(/0/);
  });

  it('relu: numerically correct [-3,-1,0,1,3]', () => {
    const t = new TensorType([5], ScalarType.F32);
    const func = buildFunction('c_relu_run', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([-3, -1, 0, 1, 3]);
    const out = new Float32Array(5);
    result.run('c_relu_run', a, out);
    expect(Array.from(out)).toEqual([0, 0, 0, 1, 3]);
  });

  it('relu(add(a,b)): fused, numerically correct', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('c_relu_add', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.relu(sum.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const a = new Float32Array([1, -5, 3, -2]);
    const b2 = new Float32Array([-4, 2, -1, -1]);
    const out = new Float32Array(4);
    result.run('c_relu_add', a, b2, out);
    expect(Array.from(out)).toEqual([0, 0, 2, 0]);
  });
});


describe('CPU kernel quality — more math numerical correctness', () => {
  it('tanh: tanh([0, 1, -1]) correct', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_tanh', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, 1, -1]);
    const out = new Float32Array(3);
    result.run('c_run_tanh', a, out);
    expect(out[0]).toBeCloseTo(Math.tanh(0), 5);
    expect(out[1]).toBeCloseTo(Math.tanh(1), 4);
    expect(out[2]).toBeCloseTo(Math.tanh(-1), 4);
  });

  it('abs: abs([-3, 0, 5]) = [3, 0, 5]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_abs', [t], [t], (b, args) => {
      b.returnOp([b.abs(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([-3, 0, 5]);
    const out = new Float32Array(3);
    result.run('c_run_abs', a, out);
    expect(Array.from(out)).toEqual([3, 0, 5]);
  });

  it('log: log([1, e, e^2]) ≈ [0, 1, 2]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('c_run_log', [t], [t], (b, args) => {
      b.returnOp([b.log(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, Math.E, Math.E * Math.E]);
    const out = new Float32Array(3);
    result.run('c_run_log', a, out);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(1, 4);
    expect(out[2]).toBeCloseTo(2, 3);
  });
});


describe('CPU kernel quality — 3D numerical correctness', () => {
  it('[2,2,3] add: correct element-wise', () => {
    const t = new TensorType([2, 2, 3], ScalarType.F32);
    const func = buildFunction('c_run_3d_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const b2 = new Float32Array(12).fill(10);
    const out = new Float32Array(12);
    result.run('c_run_3d_add', a, b2, out);
    expect(Array.from(out)).toEqual([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]);
  });

  it('[2,3,4] neg: correct element-wise', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('c_run_3d_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(24);
    for (let i = 0; i < 24; i++) a[i] = i + 1;
    const out = new Float32Array(24);
    result.run('c_run_3d_neg', a, out);
    for (let i = 0; i < 24; i++) expect(out[i]).toBe(-(i + 1));
  });
});


describe('CPU kernel quality — prod reduction', () => {
  it('prod reduce [2,3]: row products', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('c_run_rprod', [tin], [tout], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.reduce(args[0], one.getResult(0), [1], 'prod').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('c_run_rprod', a, out);
    expect(out[0]).toBeCloseTo(6, 3);
    expect(out[1]).toBeCloseTo(120, 3);
  });
});


describe('CPU kernel quality — non-divisible scheduling', () => {
  it('size 13 (not divisible by 8): still produces correct results', () => {
    const t = new TensorType([13], ScalarType.F32);
    const func = buildFunction('c_guard13', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(13).fill(3);
    const b2 = new Float32Array(13).fill(4);
    const out = new Float32Array(13);
    result.run('c_guard13', a, b2, out);
    for (let i = 0; i < 13; i++) expect(out[i]).toBe(7);
  });

  it('size 100: correct despite non-power-of-2', () => {
    const t = new TensorType([100], ScalarType.F32);
    const func = buildFunction('c_guard100', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(100);
    for (let i = 0; i < 100; i++) a[i] = i;
    const out = new Float32Array(100);
    result.run('c_guard100', a, out);
    for (let i = 0; i < 100; i++) expect(out[i]).toBe(-i);
  });

  it('[5,7] 2D non-divisible: correct', () => {
    const t = new TensorType([5, 7], ScalarType.F32);
    const func = buildFunction('c_guard_2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(35).fill(1);
    const b2 = new Float32Array(35).fill(2);
    const out = new Float32Array(35);
    result.run('c_guard_2d', a, b2, out);
    for (let i = 0; i < 35; i++) expect(out[i]).toBe(3);
  });
});


describe('CPU kernel quality — matmul epilogue numerical', () => {
  it('matmul + exp: correct exp(A@B)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_mmexp', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.exp(mm.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 0]);
    const b2 = new Float32Array([1, 0, 0, 1]);
    const o = new Float32Array(4);
    result.run('c_run_mmexp', a, b2, o);
    expect(o[0]).toBeCloseTo(Math.exp(1), 4);
    expect(o[1]).toBeCloseTo(Math.exp(0), 4);
    expect(o[2]).toBeCloseTo(Math.exp(0), 4);
    expect(o[3]).toBeCloseTo(Math.exp(0), 4);
  });

  it('matmul + tanh: correct tanh(A@B)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_mmtanh', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.tanh(mm.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([2, 0, 0, -2]);
    const o = new Float32Array(4);
    result.run('c_run_mmtanh', a, b2, o);
    expect(o[0]).toBeCloseTo(Math.tanh(2), 4);
    expect(o[1]).toBeCloseTo(Math.tanh(0), 4);
    expect(o[2]).toBeCloseTo(Math.tanh(0), 4);
    expect(o[3]).toBeCloseTo(Math.tanh(-2), 4);
  });

  it('matmul + sqrt(abs(x)): chained epilogue numerical', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_mmsqrtabs', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const ab = b.abs(mm.getResult(0));
      b.returnOp([b.sqrt(ab.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([-4, 9, 16, -25]);
    const o = new Float32Array(4);
    result.run('c_run_mmsqrtabs', a, b2, o);
    expect(o[0]).toBeCloseTo(2, 4);
    expect(o[1]).toBeCloseTo(3, 4);
    expect(o[2]).toBeCloseTo(4, 4);
    expect(o[3]).toBeCloseTo(5, 4);
  });
});


describe('CPU kernel quality — 3D reduction', () => {
  it('[2,3,4] sum axis 2: correct', () => {
    const tin = new TensorType([2, 3, 4], ScalarType.F32);
    const tout = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('c_run_3dsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [2], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(24);
    for (let i = 0; i < 24; i++) a[i] = 1;
    const out = new Float32Array(6);
    result.run('c_run_3dsum', a, out);
    for (let i = 0; i < 6; i++) expect(out[i]).toBeCloseTo(4, 5);
  });

  it('[2,3,4] sum axis 1: correct', () => {
    const tin = new TensorType([2, 3, 4], ScalarType.F32);
    const tout = new TensorType([2, 4], ScalarType.F32);
    const func = buildFunction('c_run_3dsum1', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(24);
    for (let i = 0; i < 24; i++) a[i] = 1;
    const out = new Float32Array(8);
    result.run('c_run_3dsum1', a, out);
    for (let i = 0; i < 8; i++) expect(out[i]).toBeCloseTo(3, 5);
  });
});


describe('CPU kernel quality — large matmul tiling', () => {
  it('32x64 @ 64x32: more loops than small matmul (tiling)', () => {
    const lhs = new TensorType([32, 64], ScalarType.F32);
    const rhs = new TensorType([64, 32], ScalarType.F32);
    const out = new TensorType([32, 32], ScalarType.F32);
    const func = buildFunction('c_mm_big', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm_big');
    expect(countForLoops(s)).toBeGreaterThanOrEqual(3);
  });

  it('32x64 @ 64x32: numerically correct', () => {
    const lhs = new TensorType([32, 64], ScalarType.F32);
    const rhs = new TensorType([64, 32], ScalarType.F32);
    const out = new TensorType([32, 32], ScalarType.F32);
    const func = buildFunction('c_mm_big_run', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(32 * 64).fill(1);
    const b2 = new Float32Array(64 * 32).fill(1);
    const o = new Float32Array(32 * 32);
    result.run('c_mm_big_run', a, b2, o);
    for (let i = 0; i < 32 * 32; i++) expect(o[i]).toBeCloseTo(64, 2);
  });
});


describe('CPU kernel quality — compound graph numerical', () => {
  it('matmul + bias + exp: correct exp(A@B + bias)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const bias = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('c_run_compound', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.exp(biased.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([0, 0, 0, 0]);
    const bi = new Float32Array([0, 1, 2, 0]);
    const o = new Float32Array(4);
    result.run('c_run_compound', a, b2, bi, o);
    expect(o[0]).toBeCloseTo(Math.exp(0), 4);
    expect(o[1]).toBeCloseTo(Math.exp(1), 4);
    expect(o[2]).toBeCloseTo(Math.exp(2), 4);
    expect(o[3]).toBeCloseTo(Math.exp(0), 4);
  });
});
