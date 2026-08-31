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


describe('CPU kernel quality — JS syntax, no CUDA artifacts', () => {
  it('elementwise add: uses function, not __global__', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('c_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_add');
    expect(s).toMatch(/^function\s+c_add\(/m);
    expect(s).not.toMatch(/__global__/);
    expect(s).not.toMatch(/threadIdx/);
    expect(s).not.toMatch(/blockIdx/);
    expect(s).not.toMatch(/float\*/);
  });

  it('reduction: uses function, Math.*, no CUDA', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('c_rsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'c_rsum');
    expect(s).toMatch(/^function\s+c_rsum\(/m);
    expect(s).not.toMatch(/__global__|threadIdx|blockIdx|__shared__/);
  });

  it('matmul: uses function, no CUDA', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm');
    expect(s).toMatch(/^function\s+c_mm\(/m);
    expect(s).not.toMatch(/__global__|threadIdx|blockIdx|expf|tanhf/);
  });
});


describe('CPU kernel quality — Math.* function mapping', () => {
  const mathOps = [
    { name: 'exp', jsFn: 'Math.exp', build: (b, x) => b.exp(x) },
    { name: 'log', jsFn: 'Math.log', build: (b, x) => b.log(x) },
    { name: 'sqrt', jsFn: 'Math.sqrt', build: (b, x) => b.sqrt(x) },
    { name: 'tanh', jsFn: 'Math.tanh', build: (b, x) => b.tanh(x) },
    { name: 'abs', jsFn: 'Math.abs', build: (b, x) => b.abs(x) },
    { name: 'sin', jsFn: 'Math.sin', build: (b, x) => b.sin(x) },
    { name: 'cos', jsFn: 'Math.cos', build: (b, x) => b.cos(x) },
    { name: 'floor', jsFn: 'Math.floor', build: (b, x) => b.floor(x) },
    { name: 'ceil', jsFn: 'Math.ceil', build: (b, x) => b.ceil(x) },
  ];

  for (const { name, jsFn, build } of mathOps) {
    it(`${name} maps to ${jsFn}`, () => {
      const t = new TensorType([64], ScalarType.F32);
      const func = buildFunction(`c_math_${name}`, [t], [t], (b, args) => {
        b.returnOp([build(b, args[0]).getResult(0)]);
      });
      const s = src(compile(func), `c_math_${name}`);
      expect(s).toContain(jsFn);
      expect(s).not.toMatch(/expf|logf|sqrtf|tanhf|fabsf|sinf|cosf|floorf|ceilf/);
    });
  }
});


describe('CPU kernel quality — elementwise fusion', () => {
  it('add+mul fuses into single inline expression per element', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('c_addmul', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    const s = src(compile(func), 'c_addmul');
    expect(s).toMatch(/\+.*\*/);
  });

  it('add+mul+neg: single inline expression with negation', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('c_chain3', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      b.returnOp([b.neg(prod.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_chain3');
    expect(s).toMatch(/-/);
    expect(s).toMatch(/\+/);
    expect(s).toMatch(/\*/);
  });

  it('exp(tanh(x)): nested math calls', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_exptanh', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.tanh(args[0]).getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_exptanh');
    expect(s).toMatch(/Math\.exp\(Math\.fround\(Math\.tanh\(/);
  });

  it('sqrt(abs(x)): nested math calls', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_sqrtabs', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(b.abs(args[0]).getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_sqrtabs');
    expect(s).toMatch(/Math\.sqrt\(Math\.abs\(/);
  });
});


describe('CPU kernel quality — scheduling and vectorization', () => {
  it('scheduled 1024 add: outer for loop with unrolled inner', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('c_sched_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_sched_add');
    expect(countForLoops(s)).toBeGreaterThanOrEqual(1);
    expect(s).toMatch(/\* 8/);
  });

  it('no scheduling: simple for loop, no unrolling', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('c_nosched', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compileUnscheduled(func), 'c_nosched');
    expect(countForLoops(s)).toBe(1);
    expect(s).not.toMatch(/\* 8\)/);
  });

  it('2D scheduled add: outer loop parallelized, inner vectorized', () => {
    const t = new TensorType([32, 64], ScalarType.F32);
    const func = buildFunction('c_sched_2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_sched_2d');
    expect(s).toMatch(/\* 64/);
    expect(s).toMatch(/\* 8/);
  });
});


describe('CPU kernel quality — reduction structure', () => {
  it('sum reduction: initializes to 0', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_rsum_init', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'c_rsum_init');
    expect(s).toMatch(/= 0[;,\s)]/);
  });

  it('max reduction: initializes to -Infinity', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('c_rmax_init', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const s = src(compile(func), 'c_rmax_init');
    expect(s).toMatch(/-Infinity/);
  });

  it('max reduction: uses Math.max', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('c_rmax_fn', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const s = src(compile(func), 'c_rmax_fn');
    expect(s).toMatch(/Math\.max\(/);
  });

  it('sum reduction: uses + accumulation', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('c_rsum_acc', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'c_rsum_acc');
    expect(s).toMatch(/\+/);
  });

  it('reduction has nested for loops (spatial outer, reduce inner)', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_rsum_loops', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'c_rsum_loops');
    expect(countForLoops(s)).toBeGreaterThanOrEqual(2);
  });
});


describe('CPU kernel quality — matmul structure', () => {
  it('matmul: 3 nested for loops (M, N, K)', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('c_mm_loops', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm_loops');
    expect(countForLoops(s)).toBeGreaterThanOrEqual(3);
  });

  it('matmul: has multiply-add accumulation', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_mm_fma', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm_fma');
    expect(s).toMatch(/_acc_0 \+ /);
    expect(s).toMatch(/\*/);
  });

  it('matmul: reads from both input buffers', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_mm_reads', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm_reads');
    const paramMatch = s.match(/function\s+\w+\(([^)]*)\)/);
    const params = paramMatch[1].split(',').map(p => p.trim());
    expect(params.length).toBe(3);
    expect(s).toContain(`${params[0]}[`);
    expect(s).toContain(`${params[1]}[`);
  });

  it('matmul: correct row-major strides (lhs: *K, rhs: *N)', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_mm_strides', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm_strides');
    expect(s).toMatch(/\* 16/);
    expect(s).toMatch(/\* 8/);
  });
});


describe('CPU kernel quality — epilogue fusion', () => {
  it('matmul + neg: produces negation, not identity', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_epi_neg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_epi_neg');
    expect(hasNoopStore(s)).toBe(false);
    expect(s).toMatch(/-/);
  });

  it('matmul + bias + relu: has Math.max and +', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const bias = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_epi_biasrelu', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_epi_biasrelu');
    expect(hasNoopStore(s)).toBe(false);
    expect(s).toMatch(/Math\.max\(/);
    expect(s).toMatch(/\+/);
  });

  it('matmul + exp: has Math.exp', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_epi_exp', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.exp(mm.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_epi_exp');
    expect(hasNoopStore(s)).toBe(false);
    expect(s).toMatch(/Math\.exp\(/);
  });

  it('matmul + tanh: has Math.tanh', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_epi_tanh', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.tanh(mm.getResult(0)).getResult(0)]);
    });
    const s = src(compile(func), 'c_epi_tanh');
    expect(hasNoopStore(s)).toBe(false);
    expect(s).toMatch(/Math\.tanh\(/);
  });
});


describe('CPU kernel quality — multi-dim stride correctness', () => {
  it('2D [32,64]: stride = * 64', () => {
    const t = new TensorType([32, 64], ScalarType.F32);
    const func = buildFunction('c_2d_stride', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_2d_stride');
    expect(s).toMatch(/\* 64/);
  });

  it('3D [4,8,16]: strides * 128 and * 16', () => {
    const t = new TensorType([4, 8, 16], ScalarType.F32);
    const func = buildFunction('c_3d_stride', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_3d_stride');
    expect(s).toMatch(/\* 128/);
    expect(s).toMatch(/\* 16/);
  });
});


describe('CPU kernel quality — balanced syntax', () => {
  function checkBalanced(s) {
    expect((s.match(/\{/g) || []).length).toBe((s.match(/\}/g) || []).length);
    expect((s.match(/\(/g) || []).length).toBe((s.match(/\)/g) || []).length);
    expect((s.match(/\[/g) || []).length).toBe((s.match(/\]/g) || []).length);
  }

  it('elementwise kernel: balanced braces/parens/brackets', () => {
    const t = new TensorType([512], ScalarType.F32);
    const func = buildFunction('c_bal_ew', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    checkBalanced(src(compile(func), 'c_bal_ew'));
  });

  it('reduction kernel: balanced braces/parens/brackets', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_bal_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    checkBalanced(src(compile(func), 'c_bal_red'));
  });

  it('matmul kernel: balanced braces/parens/brackets', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('c_bal_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    checkBalanced(src(compile(func), 'c_bal_mm'));
  });

  it('matmul + epilogue: balanced braces/parens/brackets', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_bal_epi', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    checkBalanced(src(compile(func), 'c_bal_epi'));
  });
});


describe('CPU kernel quality — single kernel output', () => {
  it('elementwise add: 1 kernel', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('c_1k_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });

  it('reduction: 1 kernel', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('c_1k_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });

  it('matmul + epilogue: 1 kernel', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('c_1k_epi', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });
});


describe('CPU kernel quality — no CUDA function names', () => {
  const cudaFuncs = /\b(expf|logf|sqrtf|tanhf|fabsf|sinf|cosf|floorf|ceilf|fmaxf|fminf|powf|rsqrtf)\b/;

  it('exp kernel: no CUDA funcs', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_nocuda_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    expect(src(compile(func), 'c_nocuda_exp')).not.toMatch(cudaFuncs);
  });

  it('reduction with Math.max: no CUDA funcs', () => {
    const tin = new TensorType([16, 32], ScalarType.F32);
    const tout = new TensorType([16], ScalarType.F32);
    const func = buildFunction('c_nocuda_rmax', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    expect(src(compile(func), 'c_nocuda_rmax')).not.toMatch(cudaFuncs);
  });

  it('matmul + tanh epilogue: no CUDA funcs', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('c_nocuda_mmtanh', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.tanh(mm.getResult(0)).getResult(0)]);
    });
    expect(src(compile(func), 'c_nocuda_mmtanh')).not.toMatch(cudaFuncs);
  });
});


describe('CPU kernel quality — no redundant stores', () => {
  it('neg: no noop store (buf[i] = buf[i])', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_noop_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    expect(hasNoopStore(src(compile(func), 'c_noop_neg'))).toBe(false);
  });

  it('exp: no noop store', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_noop_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    expect(hasNoopStore(src(compile(func), 'c_noop_exp'))).toBe(false);
  });

  it('add+mul chain: no noop store', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('c_noop_chain', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    expect(hasNoopStore(src(compile(func), 'c_noop_chain'))).toBe(false);
  });
});


describe('CPU kernel quality — CPU vs GPU target difference', () => {
  it('same graph, CPU uses function not __global__', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('c_vs_gpu', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_vs_gpu');
    expect(s).toMatch(/^function\s/m);
    expect(s).not.toMatch(/__global__/);
    expect(s).not.toMatch(/threadIdx/);
    expect(s).toMatch(/for\s*\(/);
  });
});


describe('CPU kernel quality — accumulator patterns', () => {
  it('sum reduction: declares accumulator with let, not const', () => {
    const tin = new TensorType([16, 32], ScalarType.F32);
    const tout = new TensorType([16], ScalarType.F32);
    const func = buildFunction('c_acc_let', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'c_acc_let');
    expect(s).toMatch(/let\s+_acc/);
  });

  it('matmul: declares accumulator with let', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('c_mm_acc', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const s = src(compile(func), 'c_mm_acc');
    expect(s).toMatch(/let\s+_acc/);
  });

  it('accumulator is updated inside loop (not just initialized)', () => {
    const tin = new TensorType([8, 16], ScalarType.F32);
    const tout = new TensorType([8], ScalarType.F32);
    const func = buildFunction('c_acc_upd', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const s = src(compile(func), 'c_acc_upd');
    expect(s).toMatch(/_acc_\d+\s*=\s*\(_acc_\d+\s*\+/);
  });
});
