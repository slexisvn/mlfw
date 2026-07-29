import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WasmTarget, CPUTarget } from '../../../src/backend/target.js';

function compile(func, opts = {}) {
  return compileGraph(func, WasmTarget(), { scheduling: { enabled: true }, ...opts });
}

function compileNoSchedule(func) {
  return compileGraph(func, WasmTarget(), { scheduling: { enabled: false } });
}

function src(result, name) {
  return result.getSource(name);
}

function countLoops(s) {
  return (s.match(/\(loop\s/g) || []).length;
}

function countBlocks(s) {
  return (s.match(/\(block\s/g) || []).length;
}


describe('WASM kernel quality — WAT module structure', () => {
  it('elementwise add: produces valid (module) with (func)', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_struct_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_struct_add');
    expect(s).toMatch(/^\(module/);
    expect(s).toMatch(/\(func\s+\(export\s+"w_struct_add"\)/);
    expect(s).toMatch(/\(memory\s+\(export\s+"memory"\)/);
  });

  it('reduction: produces valid module', () => {
    const tin = new TensorType([8, 16], ScalarType.F32);
    const tout = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_struct_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'w_struct_red');
    expect(s).toMatch(/^\(module/);
    expect(s).toMatch(/\(func\s+\(export\s+"w_struct_red"\)/);
  });

  it('matmul: produces valid module', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_struct_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_struct_mm');
    expect(s).toMatch(/^\(module/);
    expect(s).toMatch(/\(func\s+\(export\s+"w_struct_mm"\)/);
  });
});


describe('WASM kernel quality — no JS/CUDA artifacts', () => {
  it('add: no JS function/Math/let/const', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_nojs_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_nojs_add');
    expect(s).not.toMatch(/\bfunction\b/);
    expect(s).not.toMatch(/\bMath\./);
    expect(s).not.toMatch(/\blet\s+\w/);
    expect(s).not.toMatch(/(?<!\.)\bconst\s+\w/);
    expect(s).not.toMatch(/\bvar\s+\w/);
    expect(s).not.toMatch(/Float32Array/);
  });

  it('add: no CUDA __global__/threadIdx/blockIdx', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_nocuda_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_nocuda_add');
    expect(s).not.toMatch(/__global__/);
    expect(s).not.toMatch(/threadIdx/);
    expect(s).not.toMatch(/blockIdx/);
    expect(s).not.toMatch(/expf|logf|sqrtf|tanhf/);
  });

  it('exp: no JS Math.exp or CUDA expf', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_nojs_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const s = src(compile(func), 'w_nojs_exp');
    expect(s).not.toMatch(/Math\.exp/);
    expect(s).not.toMatch(/\bexpf\b/);
    expect(s).toMatch(/call \$math_exp/);
  });
});


describe('WASM kernel quality — native WASM instructions', () => {
  const nativeOps = [
    { name: 'sqrt', wasmInstr: 'sqrt', build: (b, x) => b.sqrt(x) },
    { name: 'abs', wasmInstr: 'abs', build: (b, x) => b.abs(x) },
    { name: 'floor', wasmInstr: 'floor', build: (b, x) => b.floor(x) },
    { name: 'ceil', wasmInstr: 'ceil', build: (b, x) => b.ceil(x) },
    { name: 'neg', wasmInstr: 'neg', build: (b, x) => b.neg(x) },
  ];

  for (const { name, wasmInstr, build } of nativeOps) {
    it(`${name}: uses native ${wasmInstr}, no import`, () => {
      const t = new TensorType([4], ScalarType.F32);
      const func = buildFunction(`w_native_${name}`, [t], [t], (b, args) => {
        b.returnOp([build(b, args[0]).getResult(0)]);
      });
      const s = src(compile(func), `w_native_${name}`);
      expect(s).toMatch(new RegExp(`f32\\.${wasmInstr}|f32x4\\.${wasmInstr}`));
      expect(s).not.toMatch(new RegExp(`call \\$math_${name}`));
    });
  }
});


describe('WASM kernel quality — imported math functions', () => {
  const importedOps = [
    { name: 'exp', build: (b, x) => b.exp(x) },
    { name: 'log', build: (b, x) => b.log(x) },
    { name: 'tanh', build: (b, x) => b.tanh(x) },
    { name: 'sin', build: (b, x) => b.sin(x) },
    { name: 'cos', build: (b, x) => b.cos(x) },
  ];

  for (const { name, build } of importedOps) {
    it(`${name}: imports from "math" and calls $math_${name}`, () => {
      const t = new TensorType([4], ScalarType.F32);
      const func = buildFunction(`w_import_${name}`, [t], [t], (b, args) => {
        b.returnOp([build(b, args[0]).getResult(0)]);
      });
      const s = src(compile(func), `w_import_${name}`);
      expect(s).toMatch(new RegExp(`\\(import "math" "${name}"`));
      expect(s).toMatch(new RegExp(`call \\$math_${name}`));
    });
  }
});


describe('WASM kernel quality — arithmetic instructions', () => {
  it('add: uses f32.add', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_arith_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(src(compile(func), 'w_arith_add')).toMatch(/f32\.add|f32x4\.add/);
  });

  it('sub: uses f32.sub', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_arith_sub', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });
    expect(src(compile(func), 'w_arith_sub')).toMatch(/f32\.sub|f32x4\.sub/);
  });

  it('mul: uses f32.mul', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_arith_mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    expect(src(compile(func), 'w_arith_mul')).toMatch(/f32\.mul|f32x4\.mul/);
  });
});


describe('WASM kernel quality — elementwise fusion', () => {
  it('add+mul: single kernel', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_fuse_addmul', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });

  it('add+mul: WAT contains both f32.add and f32.mul', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_fuse_ops', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    const s = src(compile(func), 'w_fuse_ops');
    expect(s).toMatch(/f32\.add|f32x4\.add/);
    expect(s).toMatch(/f32\.mul|f32x4\.mul/);
  });

  it('exp(neg(x)): contains f32.neg and call $math_exp', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_fuse_expneg', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.neg(args[0]).getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'w_fuse_expneg');
    expect(s).toMatch(/f32\.neg|f32x4\.neg/);
    expect(s).toMatch(/call \$math_exp/);
  });
});


describe('WASM kernel quality — scheduling and vectorization', () => {
  it('scheduled 16-element add: has block/loop structure', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_sched_16', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_sched_16');
    expect(s).toMatch(/\(block\s/);
    expect(s).toMatch(/\(loop\s/);
  });

  it('no scheduling: single loop, no unrolling', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_nosched', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compileNoSchedule(func), 'w_nosched');
    expect(countLoops(s)).toBe(1);
  });

  it('scheduled add: uses i32.const 4 for stride (vectorWidth=4)', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_sched_stride', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_sched_stride');
    expect(s).toMatch(/\(i32\.const 4\)/);
  });
});


describe('WASM kernel quality — reduction structure', () => {
  it('sum reduction: has f32.add for accumulation', () => {
    const tin = new TensorType([8, 16], ScalarType.F32);
    const tout = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_red_sum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'w_red_sum');
    expect(s).toContain('f32.add');
  });

  it('max reduction: uses native f32.max', () => {
    const tin = new TensorType([8, 16], ScalarType.F32);
    const tout = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_red_max', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const s = src(compile(func), 'w_red_max');
    expect(s).toMatch(/f32\.max|f32x4\.max/);
  });

  it('reduction: has nested block/loop', () => {
    const tin = new TensorType([4, 8], ScalarType.F32);
    const tout = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_red_loops', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'w_red_loops');
    expect(countLoops(s)).toBeGreaterThanOrEqual(2);
  });
});


describe('WASM kernel quality — matmul structure', () => {
  it('matmul: has 3+ nested loops', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_mm_loops', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mm_loops');
    expect(countLoops(s)).toBeGreaterThanOrEqual(3);
  });

  it('matmul: uses f32.mul and f32.add (multiply-accumulate)', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_mm_fma', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mm_fma');
    expect(s).toMatch(/f32\.mul|f32x4\.mul/);
    expect(s).toMatch(/f32\.add|f32x4\.add/);
  });

  it('matmul: uses f32.load and f32.store', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_mm_ldst', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mm_ldst');
    expect(s).toContain('f32.load');
    expect(s).toContain('f32.store');
  });
});


describe('WASM kernel quality — memory layout', () => {
  it('allocates memory pages', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('w_mem_pages', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mem_pages');
    expect(s).toMatch(/\(memory\s+\(export "memory"\)\s+\d+\s+256\)/);
  });

  it('uses i32.mul for byte offset computation', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('w_mem_offset', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mem_offset');
    expect(s).toContain('i32.mul');
  });

  it('params are i32 (memory offsets)', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_mem_params', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mem_params');
    expect(s).toMatch(/\(param i32\)\s*\(param i32\)\s*\(param i32\)/);
  });
});


describe('WASM kernel quality — balanced parentheses', () => {
  function checkBalanced(s) {
    expect((s.match(/\(/g) || []).length).toBe((s.match(/\)/g) || []).length);
  }

  it('elementwise: balanced parens', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('w_bal_ew', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    checkBalanced(src(compile(func), 'w_bal_ew'));
  });

  it('reduction: balanced parens', () => {
    const tin = new TensorType([4, 8], ScalarType.F32);
    const tout = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_bal_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    checkBalanced(src(compile(func), 'w_bal_red'));
  });

  it('matmul: balanced parens', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_bal_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    checkBalanced(src(compile(func), 'w_bal_mm'));
  });

  it('chained ops: balanced parens', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_bal_chain', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    checkBalanced(src(compile(func), 'w_bal_chain'));
  });
});


describe('WASM kernel quality — single kernel output', () => {
  it('elementwise add: 1 kernel', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_1k_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });

  it('reduction: 1 kernel', () => {
    const tin = new TensorType([4, 8], ScalarType.F32);
    const tout = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_1k_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });

  it('matmul: 1 kernel', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_1k_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });
});


describe('WASM kernel quality — elementwise numerical correctness', () => {
  it('add: [1,2,3,4] + [10,20,30,40] = [11,22,33,44]', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_run_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4]);
    const b2 = new Float32Array([10, 20, 30, 40]);
    const out = new Float32Array(4);
    result.run('w_run_add', a, b2, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 44]);
  });

  it('neg: -[1, -2, 3] = [-1, 2, -3]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, -2, 3]);
    const out = new Float32Array(3);
    result.run('w_run_neg', a, out);
    expect(Array.from(out)).toEqual([-1, 2, -3]);
  });

  it('mul: [2,3,4] * [5,6,7] = [10,18,28]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_mul', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([2, 3, 4]);
    const b2 = new Float32Array([5, 6, 7]);
    const out = new Float32Array(3);
    result.run('w_run_mul', a, b2, out);
    expect(Array.from(out)).toEqual([10, 18, 28]);
  });

  it('sub: [10,20,30] - [1,2,3] = [9,18,27]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_sub', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([10, 20, 30]);
    const b2 = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    result.run('w_run_sub', a, b2, out);
    expect(Array.from(out)).toEqual([9, 18, 27]);
  });

  it('sqrt: sqrt([4,9,16]) = [2,3,4]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_sqrt', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([4, 9, 16]);
    const out = new Float32Array(3);
    result.run('w_run_sqrt', a, out);
    expect(out[0]).toBeCloseTo(2, 5);
    expect(out[1]).toBeCloseTo(3, 5);
    expect(out[2]).toBeCloseTo(4, 5);
  });

  it('abs: abs([-3, 0, 5]) = [3, 0, 5]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_abs', [t], [t], (b, args) => {
      b.returnOp([b.abs(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([-3, 0, 5]);
    const out = new Float32Array(3);
    result.run('w_run_abs', a, out);
    expect(Array.from(out)).toEqual([3, 0, 5]);
  });

  it('exp: exp([0, 1]) ≈ [1.0, 2.718]', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, 1]);
    const out = new Float32Array(2);
    result.run('w_run_exp', a, out);
    expect(out[0]).toBeCloseTo(1.0, 5);
    expect(out[1]).toBeCloseTo(Math.E, 4);
  });

  it('tanh: tanh([0, 1, -1]) correct', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_tanh', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, 1, -1]);
    const out = new Float32Array(3);
    result.run('w_run_tanh', a, out);
    expect(out[0]).toBeCloseTo(Math.tanh(0), 5);
    expect(out[1]).toBeCloseTo(Math.tanh(1), 4);
    expect(out[2]).toBeCloseTo(Math.tanh(-1), 4);
  });

  it('log: log([1, e]) ≈ [0, 1]', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_log', [t], [t], (b, args) => {
      b.returnOp([b.log(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, Math.E]);
    const out = new Float32Array(2);
    result.run('w_run_log', a, out);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(1, 4);
  });
});


describe('WASM kernel quality — fused chain numerical correctness', () => {
  it('add+mul: (a+b)*c correct', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_run_chain', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4]);
    const b2 = new Float32Array([10, 20, 30, 40]);
    const c = new Float32Array([2, 3, 4, 5]);
    const out = new Float32Array(4);
    result.run('w_run_chain', a, b2, c, out);
    expect(Array.from(out)).toEqual([22, 66, 132, 220]);
  });

  it('neg+exp: exp(-x) correct', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_negexp', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.neg(args[0]).getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, 1]);
    const out = new Float32Array(2);
    result.run('w_run_negexp', a, out);
    expect(out[0]).toBeCloseTo(Math.exp(0), 5);
    expect(out[1]).toBeCloseTo(Math.exp(-1), 4);
  });

  it('sqrt(abs(x)): correct', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_sqrtabs', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(b.abs(args[0]).getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([-4, 9, -16]);
    const out = new Float32Array(3);
    result.run('w_run_sqrtabs', a, out);
    expect(out[0]).toBeCloseTo(2, 5);
    expect(out[1]).toBeCloseTo(3, 5);
    expect(out[2]).toBeCloseTo(4, 5);
  });
});


describe('WASM kernel quality — reduction numerical correctness', () => {
  it('sum reduce [2,3]: row sums', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_rsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('w_run_rsum', a, out);
    expect(out[0]).toBeCloseTo(6, 5);
    expect(out[1]).toBeCloseTo(15, 5);
  });

  it('max reduce [2,4]: row maxes', () => {
    const tin = new TensorType([2, 4], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_rmax', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]);
    const out = new Float32Array(2);
    result.run('w_run_rmax', a, out);
    expect(out[0]).toBe(4);
    expect(out[1]).toBe(9);
  });

  it('sum reduce axis 0 [3,2]: column sums', () => {
    const tin = new TensorType([3, 2], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_csum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('w_run_csum', a, out);
    expect(out[0]).toBeCloseTo(9, 5);
    expect(out[1]).toBeCloseTo(12, 5);
  });

  it('full sum [8]: scalar output', () => {
    const tin = new TensorType([8], ScalarType.F32);
    const tout = new TensorType([], ScalarType.F32);
    const func = buildFunction('w_run_fullsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const out = new Float32Array(1);
    result.run('w_run_fullsum', a, out);
    expect(out[0]).toBeCloseTo(36, 5);
  });
});


describe('WASM kernel quality — matmul numerical correctness', () => {
  it('identity matmul: I @ v = v', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 1], ScalarType.F32);
    const out = new TensorType([2, 1], ScalarType.F32);
    const func = buildFunction('w_run_mmid', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const eye = new Float32Array([1, 0, 0, 1]);
    const v = new Float32Array([3, 7]);
    const o = new Float32Array(2);
    result.run('w_run_mmid', eye, v, o);
    expect(o[0]).toBeCloseTo(3, 5);
    expect(o[1]).toBeCloseTo(7, 5);
  });

  it('2x3 @ 3x2: correct product', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_mm23', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b2 = new Float32Array([7, 8, 9, 10, 11, 12]);
    const o = new Float32Array(4);
    result.run('w_run_mm23', a, b2, o);
    expect(o[0]).toBeCloseTo(58, 3);
    expect(o[1]).toBeCloseTo(64, 3);
    expect(o[2]).toBeCloseTo(139, 3);
    expect(o[3]).toBeCloseTo(154, 3);
  });

  it('matmul + neg: -(A @ B) correct values', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_mmneg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([5, 3, 2, 7]);
    const o = new Float32Array(4);
    result.run('w_run_mmneg', a, b2, o);
    expect(o[0]).toBeCloseTo(-5, 5);
    expect(o[1]).toBeCloseTo(-3, 5);
    expect(o[2]).toBeCloseTo(-2, 5);
    expect(o[3]).toBeCloseTo(-7, 5);
  });

  it('matmul size variants: all-ones product = K', () => {
    const configs = [
      { M: 1, K: 4, N: 1 },
      { M: 4, K: 4, N: 4 },
      { M: 3, K: 5, N: 7 },
    ];
    for (const { M, K, N } of configs) {
      const lhs = new TensorType([M, K], ScalarType.F32);
      const rhs = new TensorType([K, N], ScalarType.F32);
      const out = new TensorType([M, N], ScalarType.F32);
      const func = buildFunction(`w_mm_${M}${K}${N}`, [lhs, rhs], [out], (b, args) => {
        b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
      });
      const result = compile(func);
      const a = new Float32Array(M * K).fill(1);
      const b2 = new Float32Array(K * N).fill(1);
      const o = new Float32Array(M * N);
      result.run(`w_mm_${M}${K}${N}`, a, b2, o);
      for (let i = 0; i < M * N; i++) expect(o[i]).toBeCloseTo(K, 3);
    }
  });
});


describe('WASM kernel quality — multi-dim numerical correctness', () => {
  it('[2,3] add: correct', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('w_run_2d_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b2 = new Float32Array([10, 20, 30, 40, 50, 60]);
    const out = new Float32Array(6);
    result.run('w_run_2d_add', a, b2, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 44, 55, 66]);
  });

  it('[2,2,3] neg: correct 3D', () => {
    const t = new TensorType([2, 2, 3], ScalarType.F32);
    const func = buildFunction('w_run_3d_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(12);
    for (let i = 0; i < 12; i++) a[i] = i + 1;
    const out = new Float32Array(12);
    result.run('w_run_3d_neg', a, out);
    for (let i = 0; i < 12; i++) expect(out[i]).toBe(-(i + 1));
  });
});


describe('WASM kernel quality — edge cases', () => {
  it('1-element tensor: compiles and runs', () => {
    const t = new TensorType([1], ScalarType.F32);
    const func = buildFunction('w_edge1', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const a = new Float32Array([5]);
    const b2 = new Float32Array([3]);
    const out = new Float32Array(1);
    result.run('w_edge1', a, b2, out);
    expect(out[0]).toBe(8);
  });

  it('non-power-of-2 size 7: correct', () => {
    const t = new TensorType([7], ScalarType.F32);
    const func = buildFunction('w_edge7', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    const b2 = new Float32Array([10, 20, 30, 40, 50, 60, 70]);
    const out = new Float32Array(7);
    result.run('w_edge7', a, b2, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 44, 55, 66, 77]);
  });

  it('large tensor 1000: compiles and produces correct boundary values', () => {
    const t = new TensorType([1000], ScalarType.F32);
    const func = buildFunction('w_edge_big', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const a = new Float32Array(1000).fill(1);
    const b2 = new Float32Array(1000).fill(2);
    const out = new Float32Array(1000);
    result.run('w_edge_big', a, b2, out);
    expect(out[0]).toBe(3);
    expect(out[999]).toBe(3);
  });
});


describe('WASM kernel quality — stride correctness', () => {
  it('2D [4,8] add: uses i32.const 8 for stride', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('w_stride_2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_stride_2d');
    expect(s).toMatch(/\(i32\.const 8\)/);
  });

  it('3D [2,4,8] add: uses i32.const 32 and i32.const 8 for strides', () => {
    const t = new TensorType([2, 4, 8], ScalarType.F32);
    const func = buildFunction('w_stride_3d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_stride_3d');
    expect(s).toMatch(/\(i32\.const 32\)/);
    expect(s).toMatch(/\(i32\.const 8\)/);
  });
});


describe('WASM kernel quality — reduce + elementwise chain', () => {
  it('sum + neg: correct negated sum', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_sumneg', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const s = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      b.returnOp([b.neg(s.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('w_run_sumneg', a, out);
    expect(out[0]).toBeCloseTo(-6, 5);
    expect(out[1]).toBeCloseTo(-15, 5);
  });
});


describe('WASM kernel quality — 3D reduction', () => {
  it('[2,3,4] sum axis 2: correct', () => {
    const tin = new TensorType([2, 3, 4], ScalarType.F32);
    const tout = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('w_run_3dsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [2], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(24).fill(1);
    const out = new Float32Array(6);
    result.run('w_run_3dsum', a, out);
    for (let i = 0; i < 6; i++) expect(out[i]).toBeCloseTo(4, 5);
  });
});


describe('WASM kernel quality — matmul + ops numerical', () => {
  it('matmul + exp: correct exp(A@B)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_mmexp', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.exp(mm.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 0]);
    const b2 = new Float32Array([1, 0, 0, 1]);
    const o = new Float32Array(4);
    result.run('w_run_mmexp', a, b2, o);
    expect(o[0]).toBeCloseTo(Math.exp(1), 4);
    expect(o[1]).toBeCloseTo(Math.exp(0), 4);
    expect(o[2]).toBeCloseTo(Math.exp(0), 4);
    expect(o[3]).toBeCloseTo(Math.exp(0), 4);
  });

  it('matmul + bias + relu: max(A@B + bias, 0)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const bias = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_biasrelu', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([3, -5, 2, 4]);
    const bi = new Float32Array([-10, 1, 1, -10]);
    const o = new Float32Array(4);
    result.run('w_run_biasrelu', a, b2, bi, o);
    expect(o[0]).toBeCloseTo(0, 5);
    expect(o[1]).toBeCloseTo(0, 5);
    expect(o[2]).toBeCloseTo(3, 5);
    expect(o[3]).toBeCloseTo(0, 5);
  });
});


describe('WASM kernel quality — relu', () => {
  it('relu: uses f32.max (native)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_relu_src', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const s = src(compile(func), 'w_relu_src');
    expect(s).toMatch(/f32\.max|f32x4\.max/);
  });

  it('relu: numerically correct [-3,-1,0,1,3]', () => {
    const t = new TensorType([5], ScalarType.F32);
    const func = buildFunction('w_relu_run', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([-3, -1, 0, 1, 3]);
    const out = new Float32Array(5);
    result.run('w_relu_run', a, out);
    expect(Array.from(out)).toEqual([0, 0, 0, 1, 3]);
  });
});


describe('WASM kernel quality — long fusion chain', () => {
  it('5-op chain (add+mul+neg+exp+sqrt): single kernel, numerically correct', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_chain5', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      const neg = b.neg(prod.getResult(0));
      const ex = b.exp(neg.getResult(0));
      b.returnOp([b.sqrt(ex.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const a = new Float32Array([1, 2]);
    const b2 = new Float32Array([1, 1]);
    const c = new Float32Array([1, 1]);
    const out = new Float32Array(2);
    result.run('w_chain5', a, b2, c, out);
    expect(out[0]).toBeCloseTo(Math.sqrt(Math.exp(-(1 + 1) * 1)), 4);
    expect(out[1]).toBeCloseTo(Math.sqrt(Math.exp(-(2 + 1) * 1)), 4);
  });
});


describe('WASM kernel quality — prod reduction', () => {
  it('prod reduce [2,3]: row products', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_rprod', [tin], [tout], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.reduce(args[0], one.getResult(0), [1], 'prod').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(2);
    result.run('w_run_rprod', a, out);
    expect(out[0]).toBeCloseTo(6, 3);
    expect(out[1]).toBeCloseTo(120, 3);
  });

  it('prod reduce: uses f32.mul for accumulation', () => {
    const tin = new TensorType([4, 8], ScalarType.F32);
    const tout = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_rprod_src', [tin], [tout], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.reduce(args[0], one.getResult(0), [1], 'prod').getResult(0)]);
    });
    const s = src(compile(func), 'w_rprod_src');
    expect(s).toMatch(/f32\.mul|f32x4\.mul/);
  });
});


describe('WASM kernel quality — control flow patterns', () => {
  it('for loop: has matching block+loop with br_if and br', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('w_cf_loop', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_cf_loop');
    expect(s).toMatch(/br_if\s+\$break_/);
    expect(s).toMatch(/br\s+\$loop_/);
  });

  it('loop labels: break label inside block, loop label inside loop', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('w_cf_labels', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_cf_labels');
    const blockLabels = s.match(/\(block\s+\$(\w+)/g) || [];
    const loopLabels = s.match(/\(loop\s+\$(\w+)/g) || [];
    expect(blockLabels.length).toBe(loopLabels.length);
  });

  it('loop counter: initialized to 0, incremented by 1, compared with ge_s', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('w_cf_counter', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const s = src(compile(func), 'w_cf_counter');
    expect(s).toContain('(i32.const 0)');
    expect(s).toContain('(i32.const 1)');
    expect(s).toContain('i32.ge_s');
    expect(s).toContain('i32.add');
  });
});


describe('WASM kernel quality — accumulator optimization', () => {
  it('sum reduction: uses _wacc local for accumulation', () => {
    const tin = new TensorType([4, 16], ScalarType.F32);
    const tout = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_wacc_sum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'w_wacc_sum');
    expect(s).toMatch(/\(local\s+\$_w?acc_\d+\s+f32\)/);
    expect(s).toMatch(/local\.set\s+\$_w?acc_/);
    expect(s).toMatch(/local\.get\s+\$_w?acc_/);
  });

  it('matmul: uses _wacc local for dot product accumulation', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('w_wacc_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_wacc_mm');
    expect(s).toMatch(/\$_w?acc_/);
  });
});


describe('WASM kernel quality — non-divisible vectorization', () => {
  it('size 5 (not divisible by 4): correct results', () => {
    const t = new TensorType([5], ScalarType.F32);
    const func = buildFunction('w_nondiv5', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5]);
    const b2 = new Float32Array([10, 20, 30, 40, 50]);
    const out = new Float32Array(5);
    result.run('w_nondiv5', a, b2, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 44, 55]);
  });

  it('size 13: correct results', () => {
    const t = new TensorType([13], ScalarType.F32);
    const func = buildFunction('w_nondiv13', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(13);
    for (let i = 0; i < 13; i++) a[i] = i + 1;
    const out = new Float32Array(13);
    result.run('w_nondiv13', a, out);
    for (let i = 0; i < 13; i++) expect(out[i]).toBe(-(i + 1));
  });

  it('[3,5] 2D non-divisible: correct', () => {
    const t = new TensorType([3, 5], ScalarType.F32);
    const func = buildFunction('w_nondiv_2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(15).fill(2);
    const b2 = new Float32Array(15).fill(3);
    const out = new Float32Array(15);
    result.run('w_nondiv_2d', a, b2, out);
    for (let i = 0; i < 15; i++) expect(out[i]).toBe(5);
  });
});


describe('WASM kernel quality — floor/ceil numerical', () => {
  it('floor: floor([1.7, -0.3, 2.0]) = [1, -1, 2]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_floor', [t], [t], (b, args) => {
      b.returnOp([b.floor(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1.7, -0.3, 2.0]);
    const out = new Float32Array(3);
    result.run('w_run_floor', a, out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(-1);
    expect(out[2]).toBe(2);
  });

  it('ceil: ceil([1.1, 2.5, 3.0]) = [2, 3, 3]', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('w_run_ceil', [t], [t], (b, args) => {
      b.returnOp([b.ceil(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1.1, 2.5, 3.0]);
    const out = new Float32Array(3);
    result.run('w_run_ceil', a, out);
    expect(out[0]).toBe(2);
    expect(out[1]).toBe(3);
    expect(out[2]).toBe(3);
  });
});


describe('WASM kernel quality — matmul + unary chain numerical', () => {
  it('matmul + tanh: correct tanh(A@B)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_mmtanh', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.tanh(mm.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([2, 0, 0, -2]);
    const o = new Float32Array(4);
    result.run('w_run_mmtanh', a, b2, o);
    expect(o[0]).toBeCloseTo(Math.tanh(2), 4);
    expect(o[1]).toBeCloseTo(Math.tanh(0), 4);
    expect(o[2]).toBeCloseTo(Math.tanh(0), 4);
    expect(o[3]).toBeCloseTo(Math.tanh(-2), 4);
  });

  it('matmul + sqrt(abs(x)): correct', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_mmsqrtabs', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const ab = b.abs(mm.getResult(0));
      b.returnOp([b.sqrt(ab.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([-4, 9, 16, -25]);
    const o = new Float32Array(4);
    result.run('w_run_mmsqrtabs', a, b2, o);
    expect(o[0]).toBeCloseTo(2, 4);
    expect(o[1]).toBeCloseTo(3, 4);
    expect(o[2]).toBeCloseTo(4, 4);
    expect(o[3]).toBeCloseTo(5, 4);
  });
});


describe('WASM kernel quality — load/store patterns', () => {
  it('add: has load and store', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_ldst_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_ldst_add');
    expect(s).toMatch(/f32\.load|v128\.load/);
    expect(s).toMatch(/f32\.store|v128\.store/);
  });

  it('neg: has load and store', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_ldst_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const s = src(compile(func), 'w_ldst_neg');
    expect(s).toMatch(/f32\.load|v128\.load/);
    expect(s).toMatch(/f32\.store|v128\.store/);
  });

  it('byte addressing: multiplies index by 4 (sizeof f32)', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_ldst_bytes', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_ldst_bytes');
    expect(s).toMatch(/\(i32\.const 4\)\s*\n\s*i32\.mul/);
  });
});


describe('WASM kernel quality — relu + add fusion', () => {
  it('relu(add(a,b)): single kernel, numerically correct', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_relu_add', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.relu(sum.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const a = new Float32Array([1, -5, 3, -2]);
    const b2 = new Float32Array([-4, 2, -1, -1]);
    const out = new Float32Array(4);
    result.run('w_relu_add', a, b2, out);
    expect(Array.from(out)).toEqual([0, 0, 2, 0]);
  });

  it('relu(add): WAT contains add and max', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('w_relu_add_src', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.relu(sum.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'w_relu_add_src');
    expect(s).toMatch(/f32\.add|f32x4\.add/);
    expect(s).toMatch(/f32\.max|f32x4\.max/);
  });
});


describe('WASM kernel quality — large matmul', () => {
  it('16x32 @ 32x16: numerically correct (all-ones = K)', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('w_mm_big', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(16 * 32).fill(1);
    const b2 = new Float32Array(32 * 16).fill(1);
    const o = new Float32Array(16 * 16);
    result.run('w_mm_big', a, b2, o);
    for (let i = 0; i < 16 * 16; i++) expect(o[i]).toBeCloseTo(32, 2);
  });

  it('large matmul: has 3+ loops', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('w_mm_big_src', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'w_mm_big_src');
    expect(countLoops(s)).toBeGreaterThanOrEqual(3);
  });
});


describe('WASM kernel quality — 3D reduction axis 1', () => {
  it('[2,3,4] sum axis 1: correct', () => {
    const tin = new TensorType([2, 3, 4], ScalarType.F32);
    const tout = new TensorType([2, 4], ScalarType.F32);
    const func = buildFunction('w_run_3dsum1', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array(24).fill(1);
    const out = new Float32Array(8);
    result.run('w_run_3dsum1', a, out);
    for (let i = 0; i < 8; i++) expect(out[i]).toBeCloseTo(3, 5);
  });
});


describe('WASM kernel quality — full max reduction', () => {
  it('max all [8]: produces single max', () => {
    const tin = new TensorType([8], ScalarType.F32);
    const tout = new TensorType([], ScalarType.F32);
    const func = buildFunction('w_fullmax', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [0], 'max').getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]);
    const out = new Float32Array(1);
    result.run('w_fullmax', a, out);
    expect(out[0]).toBe(9);
  });
});


describe('WASM kernel quality — sin/cos numerical', () => {
  it('sin: sin([0, π/2]) ≈ [0, 1]', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_sin', [t], [t], (b, args) => {
      b.returnOp([b.sin(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, Math.PI / 2]);
    const out = new Float32Array(2);
    result.run('w_run_sin', a, out);
    expect(out[0]).toBeCloseTo(0, 4);
    expect(out[1]).toBeCloseTo(1, 4);
  });

  it('cos: cos([0, π]) ≈ [1, -1]', () => {
    const t = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_cos', [t], [t], (b, args) => {
      b.returnOp([b.cos(args[0]).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, Math.PI]);
    const out = new Float32Array(2);
    result.run('w_run_cos', a, out);
    expect(out[0]).toBeCloseTo(1, 4);
    expect(out[1]).toBeCloseTo(-1, 3);
  });
});


describe('WASM kernel quality — WASM vs CPU target difference', () => {
  it('same graph: WASM uses (module), CPU uses function', () => {
    const t = new TensorType([16], ScalarType.F32);
    const func = buildFunction('w_vs_cpu', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const wasmSrc = src(compile(func), 'w_vs_cpu');
    expect(wasmSrc).toMatch(/^\(module/);
    expect(wasmSrc).not.toMatch(/^function\s/m);
    expect(wasmSrc).toMatch(/f32\.add|f32x4\.add/);
    expect(wasmSrc).not.toMatch(/\bMath\./);
  });
});


describe('WASM kernel quality — compound graph end-to-end', () => {
  it('matmul + bias + exp: correct exp(A@B + bias)', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const bias = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('w_run_compound', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.exp(biased.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b2 = new Float32Array([0, 0, 0, 0]);
    const bi = new Float32Array([0, 1, 2, 0]);
    const o = new Float32Array(4);
    result.run('w_run_compound', a, b2, bi, o);
    expect(o[0]).toBeCloseTo(Math.exp(0), 4);
    expect(o[1]).toBeCloseTo(Math.exp(1), 4);
    expect(o[2]).toBeCloseTo(Math.exp(2), 4);
    expect(o[3]).toBeCloseTo(Math.exp(0), 4);
  });

  it('reduce(sum) + exp: correct exp(sum(rows))', () => {
    const tin = new TensorType([2, 3], ScalarType.F32);
    const tout = new TensorType([2], ScalarType.F32);
    const func = buildFunction('w_run_sumexp', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const s = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      b.returnOp([b.exp(s.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const a = new Float32Array([0, 0, 0, 1, 0, 0]);
    const out = new Float32Array(2);
    result.run('w_run_sumexp', a, out);
    expect(out[0]).toBeCloseTo(Math.exp(0), 4);
    expect(out[1]).toBeCloseTo(Math.exp(1), 4);
  });
});


describe('WASM boolean ops — compile and run', () => {
  const F32 = ScalarType.F32;
  const tt = (shape) => new TensorType(shape, F32);

  it('compare + logicalAnd + logicalOr + select compiles valid WASM', () => {
    const func = buildFunction('w_bool_ops', [tt([16]), tt([16])], [tt([16])], (b, args) => {
      const gt = b.compare(args[0], args[1], 'gt').getResult(0);
      const lt = b.compare(args[0], args[1], 'lt').getResult(0);
      const anded = b.logicalAnd(gt, lt).getResult(0);
      const ored = b.logicalOr(gt, anded).getResult(0);
      b.returnOp([b.select(ored, args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const s = src(result, 'w_bool_ops');
    expect(s).toContain('(module');
    expect(s).toMatch(/i32\.and|v128\.and/);
    expect(s).toMatch(/i32\.or|v128\.or/);
  });

  it('boolean select produces correct numerical results', () => {
    const func = buildFunction('w_bool_num', [tt([4]), tt([4])], [tt([4])], (b, args) => {
      const gt = b.compare(args[0], args[1], 'gt').getResult(0);
      b.returnOp([b.select(gt, args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const a = new Float32Array([5, 1, 3, 2]);
    const bv = new Float32Array([2, 4, 3, 1]);
    const out = new Float32Array(4);
    result.run('w_bool_num', a, bv, out);
    expect(out[0]).toBe(5);
    expect(out[1]).toBe(4);
    expect(out[2]).toBe(3);
    expect(out[3]).toBe(2);
  });

  it('CSE-hoisted compare does not cause type mismatch', () => {
    const func = buildFunction('w_bool_cse', [tt([8]), tt([8])], [tt([8])], (b, args) => {
      const gt = b.compare(args[0], args[1], 'gt').getResult(0);
      const lt = b.compare(args[0], args[1], 'lt').getResult(0);
      const and1 = b.logicalAnd(gt, lt).getResult(0);
      const or1 = b.logicalOr(gt, and1).getResult(0);
      b.returnOp([b.select(or1, args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const s = src(result, 'w_bool_cse');
    expect(s).not.toContain('f32.ne');
  });

  for (const kind of ['and', 'or']) {
    for (const N of [3, 8, 16]) {
      it(`logical_${kind} mask converted to float matches oracle (N=${N}, ${N < 4 ? 'scalar' : 'SIMD'}) on cpu+wasm`, () => {
        const make = (name) => buildFunction(name, [tt([N]), tt([N]), tt([N])], [tt([N])], (b, args) => {
          const [x, lo, hi] = args;
          const c1 = b.compare(x, lo, 'gt').getResult(0);
          const c2 = b.compare(x, hi, 'lt').getResult(0);
          const comb = kind === 'and' ? b.logicalAnd(c1, c2).getResult(0) : b.logicalOr(c1, c2).getResult(0);
          b.returnOp([b.convert(comb, F32).getResult(0)]);
        });
        const pattern = [-2, 2, 7, 3, 1, 4, 0, 6];
        const x = new Float32Array(N).map((_, i) => pattern[i % 8]);
        const lo = new Float32Array(N).fill(0);
        const hi = new Float32Array(N).fill(5);
        const oracle = Array.from(x).map((v, i) => {
          const c1 = v > lo[i], c2 = v < hi[i];
          return (kind === 'and' ? (c1 && c2) : (c1 || c2)) ? 1 : 0;
        });
        for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
          const result = compileGraph(make(`bm_${kind}_${tname}_${N}`), T(), { scheduling: { enabled: true } });
          const out = new Float32Array(N);
          result.run(`bm_${kind}_${tname}_${N}`, x, lo, hi, out);
          expect(Array.from(out), `${kind} N=${N} ${tname}`).toEqual(oracle);
        }
      });
    }
  }
});


function runTyped(result, name, specs) {
  const inst = result.module.instantiate(name);
  if (inst instanceof Promise) throw new Error('Expected synchronous WASM instance');
  const { exports, memory, bufferOffsets } = inst;
  const offsets = [...bufferOffsets.values()];
  for (let i = 0; i < specs.length; i++) {
    new specs[i].ctor(memory.buffer, offsets[i], specs[i].data.length).set(specs[i].data);
  }
  exports[name](...offsets.slice(0, specs.length));
  const last = specs[specs.length - 1];
  return Array.from(new last.ctor(memory.buffer, offsets[specs.length - 1], last.data.length));
}

describe('WASM codegen — i32 numerical correctness', () => {
  it('i32 add produces correct nonzero ints', () => {
    const t = new TensorType([2, 2], ScalarType.I32);
    const func = buildFunction('w_i32_add', [t, t], [t], (b, a) => {
      b.returnOp([b.add(a[0], a[1]).getResult(0)]);
    });
    const out = runTyped(compile(func), 'w_i32_add', [
      { ctor: Int32Array, data: new Int32Array([1, 2, 3, 4]) },
      { ctor: Int32Array, data: new Int32Array([10, 20, 30, 40]) },
      { ctor: Int32Array, data: new Int32Array(4) },
    ]);
    expect(out).toEqual([11, 22, 33, 44]);
  });

  it('i32 mul, sum(axis) and matmul are correct', () => {
    const t = new TensorType([4], ScalarType.I32);
    const fm = buildFunction('w_i32_mul', [t, t], [t], (b, a) => {
      b.returnOp([b.mul(a[0], a[1]).getResult(0)]);
    });
    expect(runTyped(compile(fm), 'w_i32_mul', [
      { ctor: Int32Array, data: new Int32Array([2, 3, 4, 5]) },
      { ctor: Int32Array, data: new Int32Array([3, 4, 5, 6]) },
      { ctor: Int32Array, data: new Int32Array(4) },
    ])).toEqual([6, 12, 20, 30]);

    const ti = new TensorType([2, 3], ScalarType.I32);
    const to = new TensorType([2], ScalarType.I32);
    const fs = buildFunction('w_i32_sum', [ti], [to], (b, a) => {
      const z = b.scalarConstant(0, ScalarType.I32);
      b.returnOp([b.reduce(a[0], z.getResult(0), [1], 'sum').getResult(0)]);
    });
    expect(runTyped(compile(fs), 'w_i32_sum', [
      { ctor: Int32Array, data: new Int32Array([1, 2, 3, 4, 5, 6]) },
      { ctor: Int32Array, data: new Int32Array(2) },
    ])).toEqual([6, 15]);

    const l = new TensorType([2, 2], ScalarType.I32);
    const fmm = buildFunction('w_i32_mm', [l, l], [l], (b, a) => {
      b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
    });
    expect(runTyped(compile(fmm), 'w_i32_mm', [
      { ctor: Int32Array, data: new Int32Array([1, 2, 3, 4]) },
      { ctor: Int32Array, data: new Int32Array([1, 2, 3, 4]) },
      { ctor: Int32Array, data: new Int32Array(4) },
    ])).toEqual([7, 10, 15, 22]);
  });
});

describe('WASM codegen — f64 produces valid modules and correct values', () => {
  it('f64 add compiles and is correct', () => {
    const t = new TensorType([4], ScalarType.F64);
    const func = buildFunction('w_f64_add', [t, t], [t], (b, a) => {
      b.returnOp([b.add(a[0], a[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    expect(runTyped(result, 'w_f64_add', [
      { ctor: Float64Array, data: new Float64Array([1, 2, 3, 4]) },
      { ctor: Float64Array, data: new Float64Array([10, 20, 30, 40]) },
      { ctor: Float64Array, data: new Float64Array(4) },
    ])).toEqual([11, 22, 33, 44]);
  });

  it('f64 sum(axis) and matmul compile and are correct', () => {
    const ti = new TensorType([2, 3], ScalarType.F64);
    const to = new TensorType([2], ScalarType.F64);
    const fs = buildFunction('w_f64_sum', [ti], [to], (b, a) => {
      const z = b.scalarConstant(0, ScalarType.F64);
      b.returnOp([b.reduce(a[0], z.getResult(0), [1], 'sum').getResult(0)]);
    });
    const rs = compile(fs);
    expect(rs.succeeded).toBe(true);
    expect(runTyped(rs, 'w_f64_sum', [
      { ctor: Float64Array, data: new Float64Array([1, 2, 3, 4, 5, 6]) },
      { ctor: Float64Array, data: new Float64Array(2) },
    ])).toEqual([6, 15]);

    const l = new TensorType([2, 2], ScalarType.F64);
    const fmm = buildFunction('w_f64_mm', [l, l], [l], (b, a) => {
      b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
    });
    const rmm = compile(fmm);
    expect(rmm.succeeded).toBe(true);
    expect(runTyped(rmm, 'w_f64_mm', [
      { ctor: Float64Array, data: new Float64Array([1, 2, 3, 4]) },
      { ctor: Float64Array, data: new Float64Array([1, 2, 3, 4]) },
      { ctor: Float64Array, data: new Float64Array(4) },
    ])).toEqual([7, 10, 15, 22]);
  });

  it('f64 sqrt uses native f64 op and is correct', () => {
    const t = new TensorType([4], ScalarType.F64);
    const func = buildFunction('w_f64_sqrt', [t], [t], (b, a) => {
      b.returnOp([b.sqrt(a[0]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    expect(src(result, 'w_f64_sqrt')).toMatch(/f64\.sqrt/);
    expect(runTyped(result, 'w_f64_sqrt', [
      { ctor: Float64Array, data: new Float64Array([4, 9, 16, 25]) },
      { ctor: Float64Array, data: new Float64Array(4) },
    ])).toEqual([2, 3, 4, 5]);
  });
});

describe('WASM codegen — avg_pool2d integer divisor', () => {
  it('avg_pool2d compiles (no f32.min on integer count) and is correct', () => {
    const x = new TensorType([1, 2, 8, 8], ScalarType.F32);
    const o = new TensorType([1, 2, 4, 4], ScalarType.F32);
    const func = buildFunction('w_avgpool', [x], [o], (b, a) => {
      b.returnOp([b.pool2d(a[0], 'avg', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const xd = new Float32Array(1 * 2 * 8 * 8);
    for (let i = 0; i < xd.length; i++) xd[i] = i + 1;
    const out = new Float32Array(1 * 2 * 4 * 4);
    result.run('w_avgpool', xd, out);
    expect(out[0]).toBeCloseTo(5.5, 4);
    expect(out[1]).toBeCloseTo(7.5, 4);
    expect(out[2]).toBeCloseTo(9.5, 4);
    expect(out[3]).toBeCloseTo(11.5, 4);
  });
});

describe('WASM codegen — layer_norm affine values', () => {
  it('layer_norm with weight+bias matches reference', () => {
    const D = 16;
    const x = new TensorType([2, D], ScalarType.F32);
    const w = new TensorType([D], ScalarType.F32);
    const bb = new TensorType([D], ScalarType.F32);
    const o = new TensorType([2, D], ScalarType.F32);
    const func = buildFunction('w_ln_affine', [x, w, bb], [o], (b, a) => {
      b.returnOp([b.layernorm(a[0], a[1], a[2], -1, 1e-5).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const xd = new Float32Array(2 * D);
    for (let i = 0; i < 2 * D; i++) xd[i] = (i % D) + 1;
    const wd = new Float32Array(D).fill(2);
    const bd = new Float32Array(D).fill(0.5);
    const out = new Float32Array(2 * D);
    result.run('w_ln_affine', xd, wd, bd, out);
    const ref = (row) => {
      const m = row.reduce((s, v) => s + v, 0) / D;
      const v = row.reduce((s, val) => s + (val - m) ** 2, 0) / D;
      return row.map((val, i) => ((val - m) / Math.sqrt(v + 1e-5)) * wd[i] + bd[i]);
    };
    for (let r = 0; r < 2; r++) {
      const expected = ref(Array.from(xd.slice(r * D, r * D + D)));
      for (let i = 0; i < D; i++) expect(out[r * D + i]).toBeCloseTo(expected[i], 3);
    }
  });

  it('layer_norm with affine matches reference even without scheduling', () => {
    const D = 16;
    const x = new TensorType([4, D], ScalarType.F32);
    const w = new TensorType([D], ScalarType.F32);
    const bb = new TensorType([D], ScalarType.F32);
    const o = new TensorType([4, D], ScalarType.F32);
    const func = buildFunction('w_ln_affine_ns', [x, w, bb], [o], (b, a) => {
      b.returnOp([b.layernorm(a[0], a[1], a[2], -1, 1e-5).getResult(0)]);
    });
    const result = compileNoSchedule(func);
    expect(result.succeeded).toBe(true);
    const xd = new Float32Array(4 * D);
    for (let i = 0; i < 4 * D; i++) xd[i] = (i % 7) * 0.3 - 1;
    const wd = new Float32Array(D); for (let i = 0; i < D; i++) wd[i] = 1 + i * 0.05;
    const bd = new Float32Array(D); for (let i = 0; i < D; i++) bd[i] = i * 0.01;
    const out = new Float32Array(4 * D);
    result.run('w_ln_affine_ns', xd, wd, bd, out);
    const ref = (row) => {
      const m = row.reduce((s, v) => s + v, 0) / D;
      const v = row.reduce((s, val) => s + (val - m) ** 2, 0) / D;
      return row.map((val, i) => ((val - m) / Math.sqrt(v + 1e-5)) * wd[i] + bd[i]);
    };
    for (let r = 0; r < 4; r++) {
      const expected = ref(Array.from(xd.slice(r * D, r * D + D)));
      for (let i = 0; i < D; i++) expect(out[r * D + i]).toBeCloseTo(expected[i], 3);
    }
  });
});

describe('WASM codegen — rsqrt extern', () => {
  it('rsqrt computes 1/sqrt(x), not sqrt(x)', () => {
    const x = new TensorType([4], ScalarType.F32);
    const o = new TensorType([4], ScalarType.F32);
    const func = buildFunction('w_rsqrt', [x], [o], (b, a) => {
      b.returnOp([b._inferAndBuild('rsqrt', [a[0]]).getResult(0)]);
    });
    const result = compileNoSchedule(func);
    expect(result.succeeded).toBe(true);
    const xd = new Float32Array([1, 4, 16, 0.25]);
    const out = new Float32Array(4);
    result.run('w_rsqrt', xd, out);
    const expected = [1, 0.5, 0.25, 2];
    for (let i = 0; i < 4; i++) expect(out[i]).toBeCloseTo(expected[i], 4);
  });
});
