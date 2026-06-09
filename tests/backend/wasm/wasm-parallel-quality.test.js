import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WasmTarget } from '../../../src/backend/target.js';

const F32 = ScalarType.F32;
const PAR = WasmTarget({ numCores: 4 });
const SEQ = WasmTarget({ numCores: 1 });

function compile(func, target = PAR, opts = {}) {
  return compileGraph(func, target, { scheduling: { enabled: true }, ...opts });
}

function wat(result, name) { return result.getSource(name); }

function tt(shape) { return new TensorType(shape, F32); }

function analyzeWat(s) {
  const loops = (s.match(/\(loop\s/g) || []).length;
  const blocks = (s.match(/\(block\s/g) || []).length;
  const stores = (s.match(/(?:f32\.store|v128\.store)/g) || []).length;
  const loads = (s.match(/(?:f32\.load|v128\.load)/g) || []).length;
  const parStart = (s.match(/\$_par_start/g) || []).length;
  const parEnd = (s.match(/\$_par_end/g) || []).length;
  const opens = (s.match(/\(/g) || []).length;
  const closes = (s.match(/\)/g) || []).length;
  const lines = s.split('\n').length;
  const wasmOps = (s.match(/\b(i32|f32|v128|f32x4|i32x4)\.\w+/g) || []).length;
  return { loops, blocks, stores, loads, parStart, parEnd, opens, closes, lines, wasmOps };
}

describe('parallel WASM kernel quality — codegen structure', () => {
  it('1D add: parallel outer + vectorized inner (split)', () => {
    const t = tt([512]);
    const func = buildFunction('q_add1d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_add1d');
    const info = analyzeWat(s);
    expect(info.opens).toBe(info.closes);
    expect(info.parStart).toBeGreaterThan(0);
    expect(info.parEnd).toBeGreaterThan(0);
    expect(s).toMatch(/v128\.load|v128\.store|f32x4\.\w+/);
  });

  it('2D add: outer parallel + inner serial/vectorized', () => {
    const t = tt([64, 32]);
    const func = buildFunction('q_add2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_add2d');
    const info = analyzeWat(s);
    expect(info.opens).toBe(info.closes);
    expect(info.parStart).toBeGreaterThan(0);
    expect(info.stores).toBeGreaterThanOrEqual(1);
  });

  it('3D mul: outer parallel, inner loops handle remaining dims', () => {
    const t = tt([16, 32, 16]);
    const func = buildFunction('q_mul3d', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_mul3d');
    const info = analyzeWat(s);
    expect(info.opens).toBe(info.closes);
    expect(info.parStart).toBeGreaterThan(0);
  });

  it('reduction [128,64]: spatial parallel, reduction serial', () => {
    const tin = tt([128, 64]);
    const tout = tt([128]);
    const func = buildFunction('q_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      b.returnOp([b.reduce(args[0], zero, [1], 'sum').getResult(0)]);
    });
    const s = wat(compile(func), 'q_red');
    const info = analyzeWat(s);
    expect(info.opens).toBe(info.closes);
    expect(info.parStart).toBeGreaterThan(0);
    expect(info.loops).toBeGreaterThanOrEqual(2);
  });

  it('parallel WAT uses par_start/par_end not hardcoded bounds', () => {
    const t = tt([256]);
    const func = buildFunction('q_bounds', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_bounds');
    expect(s).toMatch(/\$_par_start/);
    expect(s).toMatch(/\$_par_end/);
    expect(s).not.toMatch(/\(i32\.const 0\)\s*\n\s*local\.set \$i0\b[^_]/);
  });

  it('sequential WAT does not emit par params', () => {
    const t = tt([256]);
    const func = buildFunction('q_seq', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const s = wat(compile(func, SEQ), 'q_seq');
    expect(s).not.toMatch(/\$_par_start/);
    expect(s).not.toMatch(/\$_par_end/);
  });
});

describe('parallel WASM kernel quality — scheduling decisions', () => {
  it('below threshold: no parallelization', () => {
    const t = tt([8]);
    const func = buildFunction('q_small', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_small');
    expect(s).not.toMatch(/\$_par_start/);
  });

  it('at threshold boundary: parallelizes', () => {
    const t = tt([16]);
    const func = buildFunction('q_boundary', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_boundary');
    expect(s).toMatch(/\$_par_start/);
  });

  it('non-power-of-2 shape: still parallelizes correctly', () => {
    const t = tt([100]);
    const func = buildFunction('q_nonpow2', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const s = wat(r, 'q_nonpow2');
    expect(s).toMatch(/\$_par_start/);
    const kernel = r.module.kernels.get('q_nonpow2');
    expect(kernel.metadata.parallel.extent).toBe(25);
  });

  it('1D reduction: no parallel (single spatial dim reduced away)', () => {
    const tin = tt([8, 32]);
    const tout = tt([8]);
    const func = buildFunction('q_red1d', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      b.returnOp([b.reduce(args[0], zero, [1], 'sum').getResult(0)]);
    });
    const s = wat(compile(func), 'q_red1d');
    expect(s).not.toMatch(/\$_par_start/);
  });

  it('parallel metadata extent matches outer loop extent after split', () => {
    const vectorWidth = 4;
    for (const size of [64, 128, 200, 512, 1024]) {
      const t = tt([size]);
      const func = buildFunction('q_ext_' + size, [t, t], [t], (b, args) => {
        b.returnOp([b.add(args[0], args[1]).getResult(0)]);
      });
      const r = compile(func);
      const kernel = r.module.kernels.get('q_ext_' + size);
      expect(kernel.metadata.parallel).toBeTruthy();
      const expectedExtent = size >= vectorWidth * 2 ? Math.floor(size / vectorWidth) : size;
      expect(kernel.metadata.parallel.extent).toBe(expectedExtent);
    }
  });

  it('2D parallel extent is the outer dim, not total elements', () => {
    const t = tt([64, 32]);
    const func = buildFunction('q_ext2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const kernel = r.module.kernels.get('q_ext2d');
    expect(kernel.metadata.parallel.extent).toBe(64);
  });

  it('1D parallel+SIMD: split produces both parallel and SIMD ops', () => {
    const t = tt([256]);
    const func = buildFunction('q_parsim', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_parsim');
    expect(s).toMatch(/\$_par_start/);
    expect(s).toMatch(/\$_par_end/);
    expect(s).toMatch(/v128\.load/);
    expect(s).toMatch(/f32x4\.add/);
    expect(s).toMatch(/v128\.store/);
  });
});

describe('parallel WASM kernel quality — WAT cleanliness', () => {
  it('no dead code: every load feeds a store or computation', () => {
    const t = tt([256]);
    const func = buildFunction('q_clean', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_clean');
    const info = analyzeWat(s);
    expect(info.loads).toBeLessThanOrEqual(info.stores * 3);
  });

  it('complex chain: mul+add+neg produces compact WAT', () => {
    const t = tt([256]);
    const func = buildFunction('q_chain', [t, t], [t], (b, args) => {
      const prod = b.mul(args[0], args[1]);
      const sum = b.add(prod.getResult(0), args[0]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const s = wat(compile(func), 'q_chain');
    const info = analyzeWat(s);
    expect(info.opens).toBe(info.closes);
    expect(info.lines).toBeLessThan(80);
  });

  it('exp+sigmoid: math imports declared, no inline bloat', () => {
    const t = tt([128]);
    const func = buildFunction('q_math', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_math');
    expect(s).toMatch(/\(import "math"/);
    const info = analyzeWat(s);
    expect(info.opens).toBe(info.closes);
  });

  it('parallel param count: buffers + shapes + 2 par params', () => {
    const t = tt([64]);
    const func = buildFunction('q_pcount', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'q_pcount');
    const params = (s.match(/\(param i32\)/g) || []).length;
    expect(params).toBe(5);
  });

  it('memory pages: minimal allocation, not over-provisioned', () => {
    const t = tt([64]);
    const func = buildFunction('q_mem', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const kernel = r.module.kernels.get('q_mem');
    expect(kernel.metadata.memoryPages).toBeLessThanOrEqual(2);
    expect(kernel.metadata.memoryPages).toBeGreaterThanOrEqual(1);
  });
});

describe('parallel WASM kernel quality — numerical correctness stress', () => {
  it('large add [4096]: every element correct', async () => {
    const t = tt([4096]);
    const func = buildFunction('q_big', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const a = new Float32Array(4096);
    const b2 = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) { a[i] = i * 0.1; b2[i] = -i * 0.05; }
    const out = new Float32Array(4096);
    await r.module.runAsync('q_big', a, b2, out);
    for (let i = 0; i < 4096; i++) {
      expect(out[i]).toBeCloseTo(i * 0.05, 3);
    }
  });

  it('non-power-of-2 [100]: boundary elements correct', async () => {
    const t = tt([100]);
    const func = buildFunction('q_np2_exec', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const a = new Float32Array(100).fill(7);
    const b2 = new Float32Array(100).fill(3);
    const out = new Float32Array(100);
    await r.module.runAsync('q_np2_exec', a, b2, out);
    for (let i = 0; i < 100; i++) expect(out[i]).toBe(10);
  });

  it('neg [1024]: sign flip on every element', async () => {
    const t = tt([1024]);
    const func = buildFunction('q_neg_big', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const a = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) a[i] = (i % 2 === 0) ? i : -i;
    const out = new Float32Array(1024);
    await r.module.runAsync('q_neg_big', a, out);
    for (let i = 0; i < 1024; i++) {
      expect(out[i]).toBe(-a[i]);
    }
  });

  it('mul [2048]: element-wise product', async () => {
    const t = tt([2048]);
    const func = buildFunction('q_mul_big', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const a = new Float32Array(2048);
    const b2 = new Float32Array(2048);
    for (let i = 0; i < 2048; i++) { a[i] = 0.5; b2[i] = i; }
    const out = new Float32Array(2048);
    await r.module.runAsync('q_mul_big', a, b2, out);
    for (let i = 0; i < 2048; i++) {
      expect(out[i]).toBeCloseTo(i * 0.5, 4);
    }
  });

  it('2D add [128,64]: row-major correctness across chunks', async () => {
    const t = tt([128, 64]);
    const func = buildFunction('q_2d_big', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const n = 128 * 64;
    const a = new Float32Array(n);
    const b2 = new Float32Array(n);
    for (let i = 0; i < n; i++) { a[i] = i; b2[i] = n - i; }
    const out = new Float32Array(n);
    await r.module.runAsync('q_2d_big', a, b2, out);
    for (let i = 0; i < n; i++) expect(out[i]).toBe(n);
  });

  it('reduction [256,32]: parallel row sums match sequential', async () => {
    const tin = tt([256, 32]);
    const tout = tt([256]);
    const func = buildFunction('q_red_big', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      b.returnOp([b.reduce(args[0], zero, [1], 'sum').getResult(0)]);
    });
    const rPar = compile(func, PAR);
    const rSeq = compile(func, SEQ);
    const a = new Float32Array(256 * 32);
    for (let i = 0; i < a.length; i++) a[i] = (i % 32) + 1;
    const outPar = new Float32Array(256);
    const outSeq = new Float32Array(256);
    await rPar.module.runAsync('q_red_big', a, outPar);
    rSeq.run('q_red_big', a, outSeq);
    for (let i = 0; i < 256; i++) {
      expect(outPar[i]).toBeCloseTo(outSeq[i], 3);
    }
  });

  it('exp chain [512]: parallel matches sequential', async () => {
    const t = tt([512]);
    const func = buildFunction('q_exp_chain', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const rPar = compile(func, PAR);
    const rSeq = compile(func, SEQ);
    const a = new Float32Array(512);
    for (let i = 0; i < 512; i++) a[i] = (i - 256) * 0.01;
    const outPar = new Float32Array(512);
    const outSeq = new Float32Array(512);
    await rPar.module.runAsync('q_exp_chain', a, outPar);
    rSeq.run('q_exp_chain', a, outSeq);
    for (let i = 0; i < 512; i++) {
      expect(outPar[i]).toBeCloseTo(outSeq[i], 4);
    }
  });
});

describe('parallel WASM kernel quality — parallel vs sequential equivalence', () => {
  it('add: async parallel output === sync sequential output', async () => {
    const t = tt([1024]);
    const func = buildFunction('q_equiv_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const rPar = compile(func, PAR);
    const rSeq = compile(func, SEQ);
    const a = new Float32Array(1024);
    const b2 = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) { a[i] = Math.sin(i); b2[i] = Math.cos(i); }
    const outPar = new Float32Array(1024);
    const outSeq = new Float32Array(1024);
    await rPar.module.runAsync('q_equiv_add', a, b2, outPar);
    rSeq.run('q_equiv_add', a, b2, outSeq);
    for (let i = 0; i < 1024; i++) {
      expect(outPar[i]).toBeCloseTo(outSeq[i], 5);
    }
  });

  it('mul+neg chain: parallel === sequential', async () => {
    const t = tt([512]);
    const func = buildFunction('q_equiv_chain', [t, t], [t], (b, args) => {
      const prod = b.mul(args[0], args[1]);
      b.returnOp([b.neg(prod.getResult(0)).getResult(0)]);
    });
    const rPar = compile(func, PAR);
    const rSeq = compile(func, SEQ);
    const a = new Float32Array(512);
    const b2 = new Float32Array(512);
    for (let i = 0; i < 512; i++) { a[i] = i * 0.01 - 2.56; b2[i] = 3.14; }
    const outPar = new Float32Array(512);
    const outSeq = new Float32Array(512);
    await rPar.module.runAsync('q_equiv_chain', a, b2, outPar);
    rSeq.run('q_equiv_chain', a, b2, outSeq);
    for (let i = 0; i < 512; i++) {
      expect(outPar[i]).toBeCloseTo(outSeq[i], 5);
    }
  });

  it('2D reduction: parallel === sequential', async () => {
    const tin = tt([64, 128]);
    const tout = tt([64]);
    const func = buildFunction('q_equiv_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      b.returnOp([b.reduce(args[0], zero, [1], 'sum').getResult(0)]);
    });
    const rPar = compile(func, PAR);
    const rSeq = compile(func, SEQ);
    const a = new Float32Array(64 * 128);
    for (let i = 0; i < a.length; i++) a[i] = (i * 7 + 13) % 100 * 0.01;
    const outPar = new Float32Array(64);
    const outSeq = new Float32Array(64);
    await rPar.module.runAsync('q_equiv_red', a, outPar);
    rSeq.run('q_equiv_red', a, outSeq);
    for (let i = 0; i < 64; i++) {
      expect(outPar[i]).toBeCloseTo(outSeq[i], 2);
    }
  });

  it('sequential fallback matches async on same parallel kernel', async () => {
    const t = tt([512]);
    const func = buildFunction('q_fallback', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    const a = new Float32Array(512);
    const b2 = new Float32Array(512);
    for (let i = 0; i < 512; i++) { a[i] = i; b2[i] = -i * 0.5; }
    const outSync = new Float32Array(512);
    const outAsync = new Float32Array(512);
    r.run('q_fallback', a, b2, outSync);
    await r.module.runAsync('q_fallback', a, b2, outAsync);
    for (let i = 0; i < 512; i++) {
      expect(outAsync[i]).toBeCloseTo(outSync[i], 5);
    }
  });
});

describe('parallel WASM kernel quality — edge cases', () => {
  it('extent == numCores: each worker gets 1 iteration', async () => {
    const t = tt([16]);
    const func = buildFunction('q_edge_min', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    const a = new Float32Array(16).fill(1);
    const b2 = new Float32Array(16).fill(2);
    const out = new Float32Array(16);
    await r.module.runAsync('q_edge_min', a, b2, out);
    for (let i = 0; i < 16; i++) expect(out[i]).toBe(3);
  });

  it('prime-sized extent [97]: uneven chunks still correct', async () => {
    const t = tt([97]);
    const func = buildFunction('q_prime', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    const a = new Float32Array(97);
    const b2 = new Float32Array(97);
    for (let i = 0; i < 97; i++) { a[i] = i; b2[i] = 97 - i; }
    const out = new Float32Array(97);
    await r.module.runAsync('q_prime', a, b2, out);
    for (let i = 0; i < 97; i++) expect(out[i]).toBe(97);
  });

  it('zeros input: output stays zero for add(0,0)', async () => {
    const t = tt([256]);
    const func = buildFunction('q_zeros', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    const a = new Float32Array(256);
    const b2 = new Float32Array(256);
    const out = new Float32Array(256).fill(999);
    await r.module.runAsync('q_zeros', a, b2, out);
    for (let i = 0; i < 256; i++) expect(out[i]).toBe(0);
  });

  it('negative values: correctly handled across chunks', async () => {
    const t = tt([512]);
    const func = buildFunction('q_negvals', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    const a = new Float32Array(512);
    const b2 = new Float32Array(512);
    for (let i = 0; i < 512; i++) { a[i] = -1; b2[i] = i; }
    const out = new Float32Array(512);
    await r.module.runAsync('q_negvals', a, b2, out);
    for (let i = 0; i < 512; i++) expect(out[i]).toBe(-i);
  });

  it('very large [8192]: no buffer overflow or corruption', async () => {
    const t = tt([8192]);
    const func = buildFunction('q_vlarge', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    const a = new Float32Array(8192);
    const b2 = new Float32Array(8192);
    for (let i = 0; i < 8192; i++) { a[i] = i; b2[i] = 1; }
    const out = new Float32Array(8192);
    await r.module.runAsync('q_vlarge', a, b2, out);
    for (let i = 0; i < 8192; i++) expect(out[i]).toBe(i + 1);
  });

  it('repeated runs: no state leak between invocations', async () => {
    const t = tt([256]);
    const func = buildFunction('q_noleak', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func, PAR);
    for (let run = 0; run < 3; run++) {
      const a = new Float32Array(256).fill(run + 1);
      const b2 = new Float32Array(256).fill(10);
      const out = new Float32Array(256);
      await r.module.runAsync('q_noleak', a, b2, out);
      for (let i = 0; i < 256; i++) expect(out[i]).toBe(run + 11);
    }
  });
});
