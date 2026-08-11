import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WasmTarget } from '../../../src/backend/target.js';
import { F32 } from '../../_utils/ir_fixture.js';

const SIMD = WasmTarget();

function compile(func, target = SIMD) {
  return compileGraph(func, target, { scheduling: { enabled: true } });
}

function wat(result, name) { return result.getSource(name); }
function tt(shape) { return new TensorType(shape, F32); }

describe('WASM SIMD — elementwise WAT output', () => {
  it('add [256] emits v128.load + f32x4.add + v128.store', () => {
    const t = tt([256]);
    const func = buildFunction('simd_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_add');
    expect(s).toContain('v128.load');
    expect(s).toContain('f32x4.add');
    expect(s).toContain('v128.store');
  });

  it('sub [128] emits f32x4.sub', () => {
    const t = tt([128]);
    const func = buildFunction('simd_sub', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_sub');
    expect(s).toContain('f32x4.sub');
  });

  it('mul [64] emits f32x4.mul', () => {
    const t = tt([64]);
    const func = buildFunction('simd_mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_mul');
    expect(s).toContain('f32x4.mul');
  });

  it('neg [256] emits f32x4.neg', () => {
    const t = tt([256]);
    const func = buildFunction('simd_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_neg');
    expect(s).toContain('f32x4.neg');
  });

  it('sqrt native emits f32x4.sqrt', () => {
    const t = tt([64]);
    const func = buildFunction('simd_sqrt', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_sqrt');
    expect(s).toContain('f32x4.sqrt');
  });

  it('abs native emits f32x4.abs', () => {
    const t = tt([64]);
    const func = buildFunction('simd_abs', [t], [t], (b, args) => {
      b.returnOp([b.abs(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_abs');
    expect(s).toContain('f32x4.abs');
  });

  it('ceil emits f32x4.ceil', () => {
    const t = tt([64]);
    const func = buildFunction('simd_ceil', [t], [t], (b, args) => {
      b.returnOp([b.ceil(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_ceil');
    expect(s).toContain('f32x4.ceil');
  });

  it('floor emits f32x4.floor', () => {
    const t = tt([64]);
    const func = buildFunction('simd_floor', [t], [t], (b, args) => {
      b.returnOp([b.floor(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_floor');
    expect(s).toContain('f32x4.floor');
  });

  it('relu zero-fill emits f32x4.splat for constant', () => {
    const t = tt([64]);
    const func = buildFunction('simd_splat', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_splat');
    expect(s).toContain('f32x4.splat');
  });
});

describe('WASM SIMD — scalarize fallback', () => {
  it('exp uses extract_lane + call $math_exp', () => {
    const t = tt([64]);
    const func = buildFunction('simd_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_exp');
    expect(s).toContain('f32x4.extract_lane');
    expect(s).toContain('call $math_exp');
  });

  it('log uses scalarize fallback', () => {
    const t = tt([64]);
    const func = buildFunction('simd_log', [t], [t], (b, args) => {
      b.returnOp([b.log(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_log');
    expect(s).toContain('f32x4.extract_lane');
    expect(s).toContain('call $math_log');
  });

  it('tanh uses scalarize fallback', () => {
    const t = tt([64]);
    const func = buildFunction('simd_tanh', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_tanh');
    expect(s).toContain('f32x4.extract_lane');
    expect(s).toContain('call $math_tanh');
  });
});

describe('WASM SIMD — compare and select', () => {
  it('relu emits SIMD max or compare+bitselect', () => {
    const t = tt([64]);
    const func = buildFunction('simd_relu', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_relu');
    expect(s).toMatch(/f32x4\.max|v128\.bitselect/);
  });
});

describe('WASM SIMD — tail handling', () => {
  it('extent=7: SIMD main loop + scalar tail', () => {
    const t = tt([7]);
    const func = buildFunction('simd_tail7', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_tail7');
    expect(s).toContain('v128.load');
    expect(s).toContain('f32x4.add');
  });

  it('extent=4: pure SIMD, no scalar tail', () => {
    const t = tt([4]);
    const func = buildFunction('simd_exact4', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_exact4');
    expect(s).toContain('v128.load');
    expect(s).toContain('f32x4.add');
    expect(s).toContain('v128.store');
  });

  it('extent=3: no SIMD (below vectorWidth)', () => {
    const t = tt([3]);
    const func = buildFunction('simd_small3', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_small3');
    expect(s).not.toContain('v128.load');
    expect(s).not.toContain('f32x4.add');
  });

  it('extent=1: scalar only', () => {
    const t = tt([1]);
    const func = buildFunction('simd_single', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_single');
    expect(s).not.toContain('v128');
  });
});

describe('WASM SIMD — reduction vectorization', () => {
  it('sum reduction [32,32] emits v128 accumulator', () => {
    const tin = tt([32, 32]);
    const tout = tt([32]);
    const func = buildFunction('simd_redsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = wat(compile(func), 'simd_redsum');
    expect(s).toContain('f32x4.add');
    expect(s).toContain('f32x4.extract_lane');
  });
});

describe('WASM SIMD — numerical correctness', () => {
  it('add [256]: element-wise correct', () => {
    const t = tt([256]);
    const func = buildFunction('simd_run_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(256);
    const b2 = new Float32Array(256);
    for (let i = 0; i < 256; i++) { a[i] = i * 0.5; b2[i] = 100 - i * 0.25; }
    const out = new Float32Array(256);
    result.run('simd_run_add', a, b2, out);
    for (let i = 0; i < 256; i++) {
      expect(out[i]).toBeCloseTo(i * 0.5 + (100 - i * 0.25), 4);
    }
  });

  it('mul [128]: element-wise correct', () => {
    const t = tt([128]);
    const func = buildFunction('simd_run_mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(128);
    const b2 = new Float32Array(128);
    for (let i = 0; i < 128; i++) { a[i] = i + 1; b2[i] = 2; }
    const out = new Float32Array(128);
    result.run('simd_run_mul', a, b2, out);
    for (let i = 0; i < 128; i++) {
      expect(out[i]).toBe((i + 1) * 2);
    }
  });

  it('neg [64]: element-wise correct', () => {
    const t = tt([64]);
    const func = buildFunction('simd_run_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(64);
    for (let i = 0; i < 64; i++) a[i] = i - 32;
    const out = new Float32Array(64);
    result.run('simd_run_neg', a, out);
    for (let i = 0; i < 64; i++) {
      expect(out[i]).toBe(-(i - 32));
    }
  });

  it('sqrt [64]: element-wise correct', () => {
    const t = tt([64]);
    const func = buildFunction('simd_run_sqrt', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(64);
    for (let i = 0; i < 64; i++) a[i] = (i + 1) * 4;
    const out = new Float32Array(64);
    result.run('simd_run_sqrt', a, out);
    for (let i = 0; i < 64; i++) {
      expect(out[i]).toBeCloseTo(Math.sqrt((i + 1) * 4), 4);
    }
  });

  it('exp [32]: element-wise correct', () => {
    const t = tt([32]);
    const func = buildFunction('simd_run_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(32);
    for (let i = 0; i < 32; i++) a[i] = (i - 16) * 0.1;
    const out = new Float32Array(32);
    result.run('simd_run_exp', a, out);
    for (let i = 0; i < 32; i++) {
      expect(out[i]).toBeCloseTo(Math.exp((i - 16) * 0.1), 3);
    }
  });

  it('relu [128]: element-wise correct', () => {
    const t = tt([128]);
    const func = buildFunction('simd_run_relu', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(128);
    for (let i = 0; i < 128; i++) a[i] = i - 64;
    const out = new Float32Array(128);
    result.run('simd_run_relu', a, out);
    for (let i = 0; i < 128; i++) {
      expect(out[i]).toBe(Math.max(0, i - 64));
    }
  });

  it('floor [64]: element-wise correct', () => {
    const t = tt([64]);
    const func = buildFunction('simd_run_floor', [t], [t], (b, args) => {
      b.returnOp([b.floor(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(64);
    for (let i = 0; i < 64; i++) a[i] = i * 0.3 - 10;
    const out = new Float32Array(64);
    result.run('simd_run_floor', a, out);
    for (let i = 0; i < 64; i++) {
      expect(out[i]).toBe(Math.floor(i * 0.3 - 10));
    }
  });

  it('ceil [64]: element-wise correct', () => {
    const t = tt([64]);
    const func = buildFunction('simd_run_ceil', [t], [t], (b, args) => {
      b.returnOp([b.ceil(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(64);
    for (let i = 0; i < 64; i++) a[i] = i * 0.3 - 10;
    const out = new Float32Array(64);
    result.run('simd_run_ceil', a, out);
    for (let i = 0; i < 64; i++) {
      expect(out[i]).toBe(Math.ceil(i * 0.3 - 10));
    }
  });

  it('tail handling [7]: add correct across boundary', () => {
    const t = tt([7]);
    const func = buildFunction('simd_run_tail', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    const b2 = new Float32Array([10, 20, 30, 40, 50, 60, 70]);
    const out = new Float32Array(7);
    result.run('simd_run_tail', a, b2, out);
    for (let i = 0; i < 7; i++) {
      expect(out[i]).toBe(a[i] + b2[i]);
    }
  });

  it('add+mul chain [256]: fused correct', () => {
    const t = tt([256]);
    const func = buildFunction('simd_run_chain', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]).getResult(0);
      b.returnOp([b.mul(sum, args[2]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(256);
    const b2 = new Float32Array(256);
    const c = new Float32Array(256);
    for (let i = 0; i < 256; i++) { a[i] = i; b2[i] = 1; c[i] = 2; }
    const out = new Float32Array(256);
    result.run('simd_run_chain', a, b2, c, out);
    for (let i = 0; i < 256; i++) {
      expect(out[i]).toBe((i + 1) * 2);
    }
  });
});

describe('WASM SIMD — reduction correctness', () => {
  it('sum reduction [8,32]: correct row sums', () => {
    const tin = tt([8, 32]);
    const tout = tt([8]);
    const func = buildFunction('simd_run_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(8 * 32);
    for (let i = 0; i < 8; i++)
      for (let j = 0; j < 32; j++)
        a[i * 32 + j] = j + 1;
    const out = new Float32Array(8);
    result.run('simd_run_red', a, out);
    const expected = 32 * 33 / 2;
    for (let i = 0; i < 8; i++) {
      expect(out[i]).toBeCloseTo(expected, 1);
    }
  });

  it('sum reduction [4,64]: correct with larger extent', () => {
    const tin = tt([4, 64]);
    const tout = tt([4]);
    const func = buildFunction('simd_run_red64', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(4 * 64);
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 64; c++)
        a[r * 64 + c] = (r + 1) * 0.1;
    const out = new Float32Array(4);
    result.run('simd_run_red64', a, out);
    for (let r = 0; r < 4; r++) {
      expect(out[r]).toBeCloseTo((r + 1) * 0.1 * 64, 2);
    }
  });
});

describe('WASM SIMD — parallel+SIMD 1D split', () => {
  const PAR_SIMD = WasmTarget({ numCores: 2 });

  function compilePar(func) {
    return compileGraph(func, PAR_SIMD, { scheduling: { enabled: true } });
  }

  it('1D add [64] with numCores=2: WAT has both parallel and SIMD', () => {
    const t = tt([64]);
    const func = buildFunction('simd_par_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compilePar(func), 'simd_par_add');
    expect(s).toMatch(/\$_par_start/);
    expect(s).toMatch(/\$_par_end/);
    expect(s).toMatch(/v128\.load/);
    expect(s).toMatch(/f32x4\.add/);
    expect(s).toMatch(/v128\.store/);
  });

  it('1D mul [256] with numCores=2: parallel+SIMD coexist', () => {
    const t = tt([256]);
    const func = buildFunction('simd_par_mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compilePar(func), 'simd_par_mul');
    expect(s).toMatch(/\$_par_start/);
    expect(s).toMatch(/f32x4\.mul/);
    expect(s).toMatch(/v128\.store/);
  });
});

describe('WASM SIMD — codegen optimizations', () => {
  it('inline: extent==lanes produces no vloop/vbreak', () => {
    const t = tt([256]);
    const func = buildFunction('simd_opt_inline', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_opt_inline');
    expect(s).not.toMatch(/vloop_/);
    expect(s).not.toMatch(/vbreak_/);
    expect(s).toMatch(/v128\.load/);
    expect(s).toMatch(/f32x4\.add/);
    expect(s).toMatch(/v128\.store/);
  });

  it('addr CSE: binary op has single addr computation for 3 accesses', () => {
    const t = tt([64]);
    const func = buildFunction('simd_opt_cse', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_opt_cse');
    expect(s).toMatch(/_vaddr_/);
    const addrMuls = (s.match(/local\.set \$_vaddr_/g) || []).length;
    expect(addrMuls).toBe(1);
  });

  it('addr CSE: unary op (1 load + 1 store) also uses cache', () => {
    const t = tt([64]);
    const func = buildFunction('simd_opt_unary', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const s = wat(compile(func), 'simd_opt_unary');
    expect(s).toMatch(/_vaddr_/);
  });
});

describe('WASM SIMD — large data stress', () => {
  it('add [1024]: all elements correct', () => {
    const t = tt([1024]);
    const func = buildFunction('simd_big_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(1024);
    const b2 = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) { a[i] = i; b2[i] = -i; }
    const out = new Float32Array(1024);
    result.run('simd_big_add', a, b2, out);
    for (let i = 0; i < 1024; i++) {
      expect(out[i]).toBe(0);
    }
  });

  it('exp+neg chain [512]: correct', () => {
    const t = tt([512]);
    const func = buildFunction('simd_big_expneg', [t], [t], (b, args) => {
      const e = b.exp(args[0]).getResult(0);
      b.returnOp([b.neg(e).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(512);
    for (let i = 0; i < 512; i++) a[i] = (i - 256) * 0.01;
    const out = new Float32Array(512);
    result.run('simd_big_expneg', a, out);
    for (let i = 0; i < 512; i++) {
      expect(out[i]).toBeCloseTo(-Math.exp((i - 256) * 0.01), 3);
    }
  });
});

describe('WASM SIMD — numeric equivalence: SIMD on == scalar (simd off), incl. non-power-of-2 remainders', () => {
  const SCALAR = WasmTarget({ simd: false });
  const I32 = ScalarType.I32;
  const T = (sh, dt = F32) => new TensorType(sh, dt);

  function runBoth(name, inTypes, build, inputs, outNumel, tol = 0) {
    const func = buildFunction(name, inTypes, [build.outT], (b, a) => { b.returnOp([build.fn(b, a).getResult(0)]); });
    const out = {};
    for (const [k, target] of [['simd', SIMD], ['scalar', SCALAR]]) {
      const r = compileGraph(func, target, { scheduling: { enabled: true } });
      const o = new Float32Array(outNumel); r.run(name, ...inputs, o); out[k] = o;
    }
    for (let i = 0; i < outNumel; i++) {
      if (tol === 0) expect(out.simd[i], `${name} idx ${i}`).toBe(out.scalar[i]);
      else expect(Math.abs(out.simd[i] - out.scalar[i]) / (1 + Math.abs(out.scalar[i])), `${name} idx ${i}`).toBeLessThan(tol);
    }
  }
  const dataF = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = Math.sin(i * 1.3) * 2; return a; };
  const dataI = (n) => { const a = new Int32Array(n); for (let i = 0; i < n; i++) a[i] = (i * 7) % 23 - 11; return a; };

  for (const n of [3, 4, 5, 7, 8, 13, 16, 17]) {
    it(`f32 elementwise chain [${n}] simd==scalar`, () => {
      runBoth(`ew_${n}`, [T([n]), T([n])], { outT: T([n]), fn: (b, a) => b.relu(b.mul(b.add(a[0], a[1]).getResult(0), a[0]).getResult(0)) }, [dataF(n), dataF(n)], n);
    });
    it(`f32 reduce-sum [${n}] simd==scalar`, () => {
      runBoth(`red_${n}`, [T([n])], { outT: T([]), fn: (b, a) => b.reduce(a[0], b.scalarConstant(0, F32).getResult(0), [0], 'sum') }, [dataF(n)], 1, 1e-5);
    });
    it(`f32 compare+select mask [${n}] simd==scalar`, () => {
      runBoth(`sel_${n}`, [T([n]), T([n])], { outT: T([n]), fn: (b, a) => b.select(b.compare(a[0], a[1], 'gt').getResult(0), a[0], b.neg(a[1]).getResult(0)) }, [dataF(n), dataF(n)], n);
    });
    it(`i32 add+mul [${n}] simd==scalar`, () => {
      runBoth(`i32_${n}`, [T([n], I32), T([n], I32)], { outT: T([n], I32), fn: (b, a) => b.add(b.mul(a[0], a[1]).getResult(0), a[0]) }, [dataI(n), dataI(n)], n);
    });
    it(`i32 max-reduce [${n}] simd==scalar`, () => {
      runBoth(`i32red_${n}`, [T([n], I32)], { outT: T([], I32), fn: (b, a) => b.reduce(a[0], b.scalarConstant(-2147483648, I32).getResult(0), [0], 'max') }, [dataI(n)], 1);
    });
  }
});
