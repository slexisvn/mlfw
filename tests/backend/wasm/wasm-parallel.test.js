import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WasmTarget } from '../../../src/backend/target.js';
import { countLoops } from '../../_utils/kernel_source.js';

const PAR_TARGET = WasmTarget({ numCores: 4 });
const SEQ_TARGET = WasmTarget({ numCores: 1 });

function compilePar(func, opts = {}) {
  return compileGraph(func, PAR_TARGET, { scheduling: { enabled: true }, ...opts });
}

function compileSeq(func, opts = {}) {
  return compileGraph(func, SEQ_TARGET, { scheduling: { enabled: true }, ...opts });
}

function src(result, name) {
  return result.getSource(name);
}


function checkBalanced(s) {
  expect((s.match(/\(/g) || []).length).toBe((s.match(/\)/g) || []).length);
}

describe('parallel WASM codegen — _par_start/_par_end params', () => {
  it('large elementwise with numCores=4: emits _par_start/_par_end locals', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_ew_par', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compilePar(func), 'wp_ew_par');
    expect(s).toMatch(/\$_par_start/);
    expect(s).toMatch(/\$_par_end/);
  });

  it('large elementwise with numCores=1: no _par_start/_par_end', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_ew_seq', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compileSeq(func), 'wp_ew_seq');
    expect(s).not.toMatch(/\$_par_start/);
    expect(s).not.toMatch(/\$_par_end/);
  });

  it('small elementwise with numCores=4: no parallel (below threshold)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('wp_ew_small', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compilePar(func), 'wp_ew_small');
    expect(s).not.toMatch(/\$_par_start/);
  });

  it('parallel loop uses _par_start as init and _par_end as bound', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_bounds', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compilePar(func), 'wp_bounds');
    expect(s).toMatch(/\(local\.get \$_par_start\)\s*\n\s*local\.set \$/);
    expect(s).toMatch(/\(local\.get \$_par_end\)\s*\n\s*i32\.ge_s/);
  });

  it('parallel WAT has extra i32 params for chunk bounds', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wp_params', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compilePar(func), 'wp_params');
    const paramMatches = s.match(/\(param i32\)/g) || [];
    expect(paramMatches.length).toBe(5);
  });

  it('parallel WAT has balanced parentheses', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_bal', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    checkBalanced(src(compilePar(func), 'wp_bal'));
  });
});

describe('parallel WASM codegen — metadata', () => {
  it('parallel kernel has parallel metadata with extent', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('wp_meta', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    const kernel = result.module.kernels.get('wp_meta');
    expect(kernel.metadata.parallel).toBeTruthy();
    expect(kernel.metadata.parallel.extent).toBeGreaterThan(0);
  });

  it('sequential kernel has no parallel metadata', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('wp_nometa', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compileSeq(func);
    const kernel = result.module.kernels.get('wp_nometa');
    expect(kernel.metadata.parallel).toBeFalsy();
  });
});

describe('parallel WASM codegen — reduction', () => {
  it('2D reduction with numCores=4: parallelizes spatial loop', () => {
    const tin = new TensorType([64, 32], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wp_red_par', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compilePar(func), 'wp_red_par');
    expect(s).toMatch(/\$_par_start/);
    expect(s).toMatch(/\$_par_end/);
  });
});

describe('parallel WASM execution — sequential fallback correctness', () => {
  it('parallel elementwise add [256]: correct with full range', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_run_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(256);
    const b2 = new Float32Array(256);
    for (let i = 0; i < 256; i++) { a[i] = i; b2[i] = 1000 - i; }
    const out = new Float32Array(256);
    result.run('wp_run_add', a, b2, out);
    for (let i = 0; i < 256; i++) {
      expect(out[i]).toBe(1000);
    }
  });

  it('parallel neg [128]: correct', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('wp_run_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(128);
    for (let i = 0; i < 128; i++) a[i] = i + 1;
    const out = new Float32Array(128);
    result.run('wp_run_neg', a, out);
    for (let i = 0; i < 128; i++) {
      expect(out[i]).toBe(-(i + 1));
    }
  });

  it('parallel 2D add [32,16]: correct', () => {
    const t = new TensorType([32, 16], ScalarType.F32);
    const func = buildFunction('wp_run_2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    const n = 32 * 16;
    const a = new Float32Array(n).fill(3);
    const b2 = new Float32Array(n).fill(7);
    const out = new Float32Array(n);
    result.run('wp_run_2d', a, b2, out);
    for (let i = 0; i < n; i++) expect(out[i]).toBe(10);
  });

  it('parallel reduction [64,16] sum: correct row sums', () => {
    const tin = new TensorType([64, 16], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wp_run_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(64 * 16).fill(1);
    const out = new Float32Array(64);
    result.run('wp_run_red', a, out);
    for (let i = 0; i < 64; i++) expect(out[i]).toBeCloseTo(16, 5);
  });

  it('parallel exp [64]: correct', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wp_run_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(64).fill(0);
    const out = new Float32Array(64);
    result.run('wp_run_exp', a, out);
    for (let i = 0; i < 64; i++) expect(out[i]).toBeCloseTo(1.0, 5);
  });
});

describe('parallel WASM execution — async parallel correctness', () => {
  it('async parallel elementwise add [256]: correct', async () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_async_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(256);
    const b2 = new Float32Array(256);
    for (let i = 0; i < 256; i++) { a[i] = i; b2[i] = 1000 - i; }
    const out = new Float32Array(256);
    await result.runAsync('wp_async_add', a, b2, out);
    for (let i = 0; i < 256; i++) {
      expect(out[i]).toBe(1000);
    }
  });

  it('async parallel neg [128]: correct', async () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('wp_async_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(128);
    for (let i = 0; i < 128; i++) a[i] = i + 1;
    const out = new Float32Array(128);
    await result.runAsync('wp_async_neg', a, out);
    for (let i = 0; i < 128; i++) {
      expect(out[i]).toBe(-(i + 1));
    }
  });

  it('async parallel 2D [32,16] add: correct', async () => {
    const t = new TensorType([32, 16], ScalarType.F32);
    const func = buildFunction('wp_async_2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    const n = 32 * 16;
    const a = new Float32Array(n).fill(5);
    const b2 = new Float32Array(n).fill(8);
    const out = new Float32Array(n);
    await result.runAsync('wp_async_2d', a, b2, out);
    for (let i = 0; i < n; i++) expect(out[i]).toBe(13);
  });

  it('async parallel reduction [64,16] sum: correct', async () => {
    const tin = new TensorType([64, 16], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wp_async_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(64 * 16).fill(2);
    const out = new Float32Array(64);
    await result.runAsync('wp_async_red', a, out);
    for (let i = 0; i < 64; i++) expect(out[i]).toBeCloseTo(32, 4);
  });

  it('async parallel isAsync returns true for parallel kernels', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_isasync', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    expect(result.isAsync('wp_isasync')).toBe(true);
  });

  it('sequential kernel isAsync returns false', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wp_notasync', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compileSeq(func);
    expect(result.isAsync('wp_notasync')).toBe(false);
  });
});

describe('parallel WASM execution — large data parallel', () => {
  it('async parallel 1024-element add: all correct', async () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('wp_big_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(1024);
    const b2 = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) { a[i] = i * 0.5; b2[i] = i * 0.25; }
    const out = new Float32Array(1024);
    await result.runAsync('wp_big_add', a, b2, out);
    for (let i = 0; i < 1024; i++) {
      expect(out[i]).toBeCloseTo(i * 0.75, 4);
    }
  });

  it('async parallel mul+exp chain [512]: correct', async () => {
    const t = new TensorType([512], ScalarType.F32);
    const func = buildFunction('wp_big_chain', [t, t], [t], (b, args) => {
      const prod = b.mul(args[0], args[1]);
      b.returnOp([b.exp(prod.getResult(0)).getResult(0)]);
    });
    const result = compilePar(func);
    const a = new Float32Array(512).fill(0);
    const b2 = new Float32Array(512).fill(1);
    const out = new Float32Array(512);
    await result.runAsync('wp_big_chain', a, b2, out);
    for (let i = 0; i < 512; i++) {
      expect(out[i]).toBeCloseTo(1.0, 5);
    }
  });
});

// Regression tests for the scheduled-WASM correctness bugs. Before the fixes the worker
// pool partitioned every buffer contiguously by extent (wrong for reductions where the
// input is read strided, and for matmul where an operand is read in full), and the SIMD
// codegen vector-loaded strided operands of a reduction/accumulator as if contiguous.
describe('parallel WASM execution — scheduled reduction over non-last axis', () => {
  // Reduce over axis 0 (sum down columns): the parallel loop iterates the spatial
  // (column) axis while the input is read strided down rows. The pool used to hand each
  // worker a contiguous row-slice instead of its columns; now reductions run sync.
  function buildColSum(name, R, C) {
    const tin = new TensorType([R, C], ScalarType.F32);
    const tout = new TensorType([C], ScalarType.F32);
    return buildFunction(name, [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
  }

  it('sum over axis 0 [32,128] is correct (sync)', () => {
    const R = 32, C = 128;
    const a = new Float32Array(R * C);
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) a[i * C + j] = (j + 1) * 0.01;
    const out = new Float32Array(C);
    compilePar(buildColSum('wp_colsum_s', R, C)).run('wp_colsum_s', a, out);
    for (let j = 0; j < C; j++) expect(out[j]).toBeCloseTo(R * (j + 1) * 0.01, 3);
  });

  it('sum over axis 0 [32,128] is correct (async/pool path)', async () => {
    const R = 32, C = 128;
    const a = new Float32Array(R * C);
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) a[i * C + j] = (j + 1) * 0.01;
    const out = new Float32Array(C);
    await compilePar(buildColSum('wp_colsum_a', R, C)).runAsync('wp_colsum_a', a, out);
    for (let j = 0; j < C; j++) expect(out[j]).toBeCloseTo(R * (j + 1) * 0.01, 3);
  });

  it('max over axis 0 [16,64] is correct (vectorized reduction)', () => {
    const R = 16, C = 64;
    const a = new Float32Array(R * C);
    // column j max occurs at row j % R with value j + 1; other rows smaller.
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) a[i * C + j] = (i === j % R) ? (j + 1) : -(j + 1);
    const tin = new TensorType([R, C], ScalarType.F32);
    const tout = new TensorType([C], ScalarType.F32);
    const func = buildFunction('wp_colmax', [tin], [tout], (b, args) => {
      const init = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], init.getResult(0), [0], 'max').getResult(0)]);
    });
    const out = new Float32Array(C);
    compilePar(func).run('wp_colmax', a, out);
    for (let j = 0; j < C; j++) expect(out[j]).toBeCloseTo(j + 1, 4);
  });
});

describe('parallel WASM execution — scheduled matmul (vectorized contraction)', () => {
  // matmul parallelizes over output rows and vectorizes the K contraction; operand B is
  // read with a row stride (B[k, j]). The SIMD accumulator used to vector-load B as if
  // contiguous; now it falls back to the scalar accumulator for strided operands.
  function buildMatmul(name, Mn, K, P) {
    const ta = new TensorType([Mn, K], ScalarType.F32);
    const tb = new TensorType([K, P], ScalarType.F32);
    const tc = new TensorType([Mn, P], ScalarType.F32);
    return buildFunction(name, [ta, tb], [tc], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
  }

  it('[64,8]@[8,16] matches hand-computed product (sync)', () => {
    const Mn = 64, K = 8, P = 16;
    const a = new Float32Array(Mn * K).fill(1);            // A[i,k] = 1
    const b = new Float32Array(K * P);
    for (let k = 0; k < K; k++) for (let j = 0; j < P; j++) b[k * P + j] = (j + 1) * 0.25;
    const out = new Float32Array(Mn * P);
    compilePar(buildMatmul('wp_mm_s', Mn, K, P)).run('wp_mm_s', a, b, out);
    for (let i = 0; i < Mn; i++) for (let j = 0; j < P; j++) {
      expect(out[i * P + j]).toBeCloseTo(K * (j + 1) * 0.25, 4); // sum_k 1 * (j+1)*0.25
    }
  });

  it('[64,8]@[8,16] matches hand-computed product (async/pool path)', async () => {
    const Mn = 64, K = 8, P = 16;
    const a = new Float32Array(Mn * K).fill(1);
    const b = new Float32Array(K * P);
    for (let k = 0; k < K; k++) for (let j = 0; j < P; j++) b[k * P + j] = (j + 1) * 0.25;
    const out = new Float32Array(Mn * P);
    await compilePar(buildMatmul('wp_mm_a', Mn, K, P)).runAsync('wp_mm_a', a, b, out);
    for (let i = 0; i < Mn; i++) for (let j = 0; j < P; j++) {
      expect(out[i * P + j]).toBeCloseTo(K * (j + 1) * 0.25, 4);
    }
  });
});

describe('parallel WASM execution — fused matmul->reduce with mismatched parallel extents', () => {
  // The matmul row loop (extent M) and the reduce output loop (extent N) were both marked
  // PARALLEL and both bound to the shared _par_start/_par_end partition. When M != N the
  // smaller loop over-iterated, writing one element past its buffer into the adjacent
  // const-init buffer (manifesting as a uniform constant added to every reduce output).
  // Each parallel loop now consumes the partition only when its extent matches the
  // partitioned axis; mismatched-extent parallel loops emit a plain serial loop.
  function buildMatmulReduce(name, M, K, N) {
    const ta = new TensorType([M, K], ScalarType.F32);
    const tb = new TensorType([K, N], ScalarType.F32);
    const tout = new TensorType([N], ScalarType.F32);
    return buildFunction(name, [ta, tb], [tout], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(mm, zero.getResult(0), [0], 'sum').getResult(0)]);
    });
  }

  function refMatmulColSum(A, B, M, K, N) {
    const ref = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let i = 0; i < M; i++) { let m = 0; for (let k = 0; k < K; k++) m += A[i * K + k] * B[k * N + j]; s += m; }
      ref[j] = s;
    }
    return ref;
  }

  it('matmul row-extent == vector-width but != reduce output-extent stays correct', () => {
    const M = 4, K = 2, N = 5;
    const A = new Float32Array(M * K); for (let i = 0; i < A.length; i++) A[i] = i + 1;
    const B = new Float32Array(K * N); for (let i = 0; i < B.length; i++) B[i] = (i % 3) + 1;
    const out = new Float32Array(N);
    compilePar(buildMatmulReduce('wp_mmr_45', M, K, N)).run('wp_mmr_45', A, B, out);
    const ref = refMatmulColSum(A, B, M, K, N);
    for (let j = 0; j < N; j++) expect(out[j]).toBeCloseTo(ref[j], 4);
  });

  it('mismatched M/N grid matches reference (sync)', () => {
    let n = 0;
    for (const M of [3, 4, 5, 6, 8]) for (const N of [3, 5, 6, 8, 12]) for (const K of [1, 2, 3]) {
      const A = new Float32Array(M * K); for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 1.3) * 2;
      const B = new Float32Array(K * N); for (let i = 0; i < B.length; i++) B[i] = Math.cos(i * 0.9) * 2;
      const out = new Float32Array(N);
      compilePar(buildMatmulReduce('wp_mmr_' + (n++), M, K, N)).run('wp_mmr_' + (n - 1), A, B, out);
      const ref = refMatmulColSum(A, B, M, K, N);
      for (let j = 0; j < N; j++) expect(out[j], `M=${M} N=${N} K=${K} col ${j}`).toBeCloseTo(ref[j], 3);
    }
  });
});
