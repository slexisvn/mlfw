import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WasmTarget } from '../../../src/backend/target.js';

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

function countLoops(s) {
  return (s.match(/\(loop\s/g) || []).length;
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
