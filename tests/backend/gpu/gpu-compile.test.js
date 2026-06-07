import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { GPUTarget, CPUTarget } from '../../../src/backend/target.js';
import {
  tensor, Linear, Sequential, ReLU, Sigmoid, Tanh,
  GELU, SiLU, LeakyReLU,
  compile as modelCompile,
} from '../../../src/index.js';

function compile(func, opts = {}) {
  return compileGraph(func, GPUTarget(), { scheduling: { enabled: true }, ...opts });
}

function compileNoSchedule(func, opts = {}) {
  return compileGraph(func, GPUTarget(), { scheduling: { enabled: false }, ...opts });
}

function getSource(result, name) {
  return result.getSource(name);
}

function countForLoops(src) {
  return (src.match(/\bfor\s*\(/g) || []).length;
}

function hasThreadBinding(src, tag) {
  return src.includes(`= ${tag};`) || src.includes(`= ${tag}`);
}

function extractDeclaredVars(src) {
  const declared = new Set();
  const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
  if (paramMatch) {
    for (const p of paramMatch[1].split(',')) {
      const name = p.trim().split(/\s+/).pop().replace('*', '');
      if (name) declared.add(name);
    }
  }
  for (const m of src.matchAll(/(?:const\s+int|int|float|double|__half)\s+(\w+)\s*(?:=|\[)/g)) {
    declared.add(m[1]);
  }
  for (const m of src.matchAll(/for\s*\(\s*int\s+(\w+)/g)) {
    declared.add(m[1]);
  }
  declared.add('blockIdx');
  declared.add('threadIdx');
  declared.add('INFINITY');
  return declared;
}

function findUndeclaredVars(src) {
  const declared = extractDeclaredVars(src);
  const bodyMatch = src.match(/\{([\s\S]*)\}\s*$/);
  if (!bodyMatch) return [];
  const body = bodyMatch[1];

  const dotMembers = new Set();
  for (const m of body.matchAll(/(\w+)\.(\w+)/g)) {
    dotMembers.add(`${m.index + m[1].length}`);
  }

  const used = new Set();
  for (const m of body.matchAll(/\b([a-zA-Z_]\w*)\b/g)) {
    const name = m[1];
    if (/^(const|int|float|double|void|for|if|else|while|return|__global__|__shared__|__half|__syncthreads|pragma|unroll|INFINITY|alloca|sizeof)$/.test(name)) continue;
    if (/^(expf|logf|sqrtf|tanhf|fabsf|sinf|cosf|ceilf|floorf|fmaxf|fminf|powf|roundf|fmodf|rsqrtf|rsqrt|exp|log|sqrt|tanh|fabs|sin|cos|ceil|floor|fmax|fmin|pow|round|fmod)$/.test(name)) continue;
    if (name.length <= 1 && /^[xyzw]$/.test(name)) {
      const before = m.index > 0 ? body[m.index - 1] : '';
      if (before === '.') continue;
    }
    used.add(name);
  }

  const undeclared = [];
  for (const v of used) {
    if (!declared.has(v)) undeclared.push(v);
  }
  return undeclared;
}

function countStoreStatements(src) {
  return (src.match(/\w+\[.*\]\s*=/g) || []).length;
}

function hasNoopStore(src) {
  const bodyMatch = src.match(/\{([\s\S]*)\}\s*$/);
  if (!bodyMatch) return false;
  const lines = bodyMatch[1].split('\n').map(l => l.trim()).filter(l => l.length > 0);
  for (const line of lines) {
    const m = line.match(/^(\w+\[[^\]]+\])\s*=\s*(.+);$/);
    if (m) {
      const lhs = m[1].replace(/\s+/g, '');
      const rhs = m[2].replace(/\s+/g, '');
      if (lhs === rhs) return true;
    }
  }
  return false;
}

// ────────────────────────────────────────────────────────────────────
// Elementwise: fusion quality
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — elementwise fusion', () => {
  it('add+mul fuses into single inline expression, no temp buffer', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('q_add_mul', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_add_mul');
    const kernels = result.listKernels();
    expect(kernels).toHaveLength(1);
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/\+.*\*/);
    expect(src).not.toMatch(/float\s+\w+\[\d+\]/);
  });

  it('add+mul+neg chain computes (-((a+b)*c)) in one store', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_chain3', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      b.returnOp([b.neg(prod.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_chain3');
    expect(countStoreStatements(src)).toBe(1);
    const storeMatch = src.match(/buf_\w+\[.*?\]\s*=\s*(.+);/);
    expect(storeMatch).not.toBeNull();
    const rhs = storeMatch[1];
    expect(rhs).toContain('-');
    expect(rhs).toContain('+');
    expect(rhs).toContain('*');
  });

  it('sub+neg fuses into single expression', () => {
    const t = new TensorType([512], ScalarType.F32);
    const func = buildFunction('q_sub_neg', [t, t], [t], (b, args) => {
      const diff = b.sub(args[0], args[1]);
      b.returnOp([b.neg(diff.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_sub_neg');
    expect(result.listKernels()).toHaveLength(1);
    expect(countStoreStatements(src)).toBe(1);
  });

  it('mul+exp fuses expf(a*b) in single store', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_mul_exp', [t, t], [t], (b, args) => {
      const prod = b.mul(args[0], args[1]);
      b.returnOp([b.exp(prod.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_mul_exp');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/expf\(.+\*.+\)/);
  });

  it('neg(neg(x)) algebraically simplified to identity copy', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_double_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(b.neg(args[0]).getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_double_neg');
    expect(src).not.toMatch(/\(-\(-/);
    expect(countStoreStatements(src)).toBe(1);
  });
});

// ────────────────────────────────────────────────────────────────────
// Elementwise: no temp buffers, no redundant stores
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — no waste in elementwise', () => {
  it('single unary op has exactly 1 store, 0 local arrays', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_neg');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).not.toMatch(/float\s+\w+\[\d+\]/);
  });

  it('single binary op has exactly 1 store, 0 local arrays', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_add');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).not.toMatch(/float\s+\w+\[\d+\]/);
  });

  it('exp has exactly 1 store with expf call', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_exp');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/=\s*expf\(/);
  });

  it('tanh has exactly 1 store with tanhf call', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_tanh', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_tanh');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/=\s*tanhf\(/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Thread binding: scheduling effectiveness
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — thread parallelism', () => {
  it('1024 elements: blockIdx.x + threadIdx.x, block size 256', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('q_par1k', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_par1k');
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
    expect(src).toMatch(/\* 256/);
    expect(countForLoops(src)).toBe(0);
  });

  it('128 elements: threadIdx.x only, no blockIdx split', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('q_par128', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_par128');
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(false);
    expect(countForLoops(src)).toBe(0);
  });

  it('1M elements: still uses blockIdx+threadIdx split', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('q_par1m', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_par1m');
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
    expect(countForLoops(src)).toBe(0);
  });

  it('2D tensor fuses dims before thread binding', () => {
    const t = new TensorType([32, 64], ScalarType.F32);
    const func = buildFunction('q_fuse2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, 'q_fuse2d');
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
    expect(src).toMatch(/\/ 64/);
    expect(src).toMatch(/% 64/);
    expect(countForLoops(src)).toBe(0);
  });

  it('no scheduling: plain for loops, no thread bindings', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('q_nosched', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compileNoSchedule(func);
    const src = getSource(result, 'q_nosched');
    expect(countForLoops(src)).toBeGreaterThanOrEqual(1);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(false);
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────
// Variable declaration: no undeclared vars in elementwise
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — all variables declared (elementwise)', () => {
  it('add 1D: no undeclared vars', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('q_decl_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_add');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('add 2D: no undeclared vars', () => {
    const t = new TensorType([32, 64], ScalarType.F32);
    const func = buildFunction('q_decl_add2d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_add2d');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('add+mul chain: no undeclared vars', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_decl_chain', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_chain');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('neg: no undeclared vars', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_decl_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_neg');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('exp: no undeclared vars', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_decl_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_exp');
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

// ────────────────────────────────────────────────────────────────────
// Variable declaration: reduction kernels
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — all variables declared (reduction)', () => {
  it('sum reduce axis 1: no undeclared vars', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('q_decl_rsum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_rsum');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('max reduce: no undeclared vars', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_decl_rmax', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_rmax');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('sum reduce axis 0 (col sum): no undeclared vars', () => {
    const tin = new TensorType([64, 32], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_decl_csum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_csum');
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

// ────────────────────────────────────────────────────────────────────
// Variable declaration: matmul kernels
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — all variables declared (matmul)', () => {
  it('matmul 16x32 @ 32x16: no undeclared vars', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_decl_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_mm');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('large matmul 128x128: no undeclared vars', () => {
    const t = new TensorType([128, 128], ScalarType.F32);
    const func = buildFunction('q_decl_mm128', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_mm128');
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

// ────────────────────────────────────────────────────────────────────
// Variable declaration: epilogue-fused kernels
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — all variables declared (epilogue)', () => {
  it('matmul + bias add: no undeclared vars', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const bias = new TensorType([16, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_decl_mmbias', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.add(mm.getResult(0), args[2]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_mmbias');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('matmul + neg: no undeclared vars', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_decl_mmneg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_mmneg');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('reduce + neg: no undeclared vars', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_decl_sumneg', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const sum = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_decl_sumneg');
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

// ────────────────────────────────────────────────────────────────────
// No-op stores: matmul+neg should negate, not copy
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — no redundant stores', () => {
  it('matmul + neg epilogue must negate, not produce x=x', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_noop_mmneg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_noop_mmneg');
    expect(hasNoopStore(src)).toBe(false);
  });

  it('matmul + bias epilogue must add, not produce x=x', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const bias = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_noop_mmbias', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.add(mm.getResult(0), args[2]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_noop_mmbias');
    expect(hasNoopStore(src)).toBe(false);
  });

  it('elementwise add: no noop stores', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_noop_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_noop_add');
    expect(hasNoopStore(src)).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────
// Source sanity: no JS artifacts, balanced braces
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — source sanity', () => {
  it('no JavaScript in CUDA source', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_nojs', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_nojs');
    expect(src).not.toMatch(/Math\./);
    expect(src).not.toMatch(/\blet\s/);
    expect(src).not.toMatch(/\bfunction\s/);
    expect(src).not.toMatch(/=>/);
    expect(src).not.toMatch(/Float32Array/);
    expect(src).not.toContain('undefined');
    expect(src).not.toContain('NaN');
  });

  it('balanced braces in add+exp kernel', () => {
    const t = new TensorType([512], ScalarType.F32);
    const func = buildFunction('q_braces', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.exp(sum.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_braces');
    expect((src.match(/\{/g) || []).length).toBe((src.match(/\}/g) || []).length);
  });

  it('balanced parens in nested expression', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_parens', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_parens');
    expect((src.match(/\(/g) || []).length).toBe((src.match(/\)/g) || []).length);
  });
});

// ────────────────────────────────────────────────────────────────────
// Dtype correctness in generated code
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — dtype correctness', () => {
  it('f32: float* params, expf suffix', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('q_dtype_f32', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_dtype_f32');
    expect(src).toMatch(/float\*/);
    expect(src).toMatch(/expf\(/);
    expect(src).not.toMatch(/double\*/);
  });

  it('f64: double* params, exp without suffix', () => {
    const t = new TensorType([64], ScalarType.F64);
    const func = buildFunction('q_dtype_f64', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_dtype_f64');
    expect(src).toMatch(/double\*/);
    expect(src).toMatch(/\bexp\(/);
    expect(src).not.toMatch(/expf\(/);
  });

  it('i32: int* params', () => {
    const t = new TensorType([64], ScalarType.I32);
    const func = buildFunction('q_dtype_i32', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_dtype_i32');
    expect(src).toMatch(/int\*/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Matmul kernel structure
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — matmul structure', () => {
  it('matmul initializes output to 0 before accumulation', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_mm_init', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_init');
    expect(src).toMatch(/= 0f/);
  });

  it('matmul has multiply-add accumulation pattern', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_mm_fma', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_fma');
    expect(src).toMatch(/\+ \(/);
    expect(src).toMatch(/\*/);
  });

  it('matmul reads from both input buffers', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_mm_reads', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_reads');
    const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
    const params = paramMatch[1].split(',').map(p => p.trim().split(/\s+/).pop().replace('*', ''));
    const lhsBuf = params[0];
    const rhsBuf = params[1];
    expect(src).toContain(`${lhsBuf}[`);
    expect(src).toContain(`${rhsBuf}[`);
  });
});

// ────────────────────────────────────────────────────────────────────
// Reduction kernel structure
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — reduction structure', () => {
  it('sum reduction initializes to 0', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_rsum_init', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rsum_init');
    expect(src).toMatch(/= 0f/);
  });

  it('max reduction initializes to -INFINITY', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_rmax_init', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rmax_init');
    expect(src).toMatch(/(-INFINITY)|(-Inf)/i);
  });

  it('max reduction uses fmaxf', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_rmax_func', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rmax_func');
    expect(src).toMatch(/fmaxf\(/);
  });

  it('sum reduction uses + accumulation', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_rsum_acc', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rsum_acc');
    const hasInlineAcc = /\w+\[.*?\]\s*=\s*\(.+\+.+\)/.test(src);
    const hasLIRAcc = /_acc_\d+\s*=\s*\(_acc_\d+\s*\+/.test(src);
    expect(hasInlineAcc || hasLIRAcc).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────
// GPU vs CPU codegen comparison
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — GPU vs CPU codegen differences', () => {
  it('GPU: __global__, float*; CPU: function, no __global__', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_cmp_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const gpuSrc = getSource(compile(func), 'q_cmp_add');
    const cpuSrc = getSource(compileGraph(func, CPUTarget()), 'q_cmp_add');

    expect(gpuSrc).toMatch(/__global__/);
    expect(gpuSrc).toMatch(/float\*/);
    expect(gpuSrc).not.toMatch(/Float32Array/);

    expect(cpuSrc).toMatch(/function/);
    expect(cpuSrc).not.toMatch(/__global__/);
  });

  it('GPU: expf; CPU: Math.exp', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_cmp_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });
    const gpuSrc = getSource(compile(func), 'q_cmp_exp');
    const cpuSrc = getSource(compileGraph(func, CPUTarget()), 'q_cmp_exp');

    expect(gpuSrc).toMatch(/expf\(/);
    expect(gpuSrc).not.toMatch(/Math\.exp/);
    expect(cpuSrc).toMatch(/Math\.exp\(/);
    expect(cpuSrc).not.toMatch(/expf/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Edge cases
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — edge cases', () => {
  it('1-element tensor: compiles without div-by-zero', () => {
    const t = new TensorType([1], ScalarType.F32);
    const func = buildFunction('q_edge1', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, 'q_edge1');
    expect(src).not.toContain('undefined');
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('non-power-of-2 sizes compile with all vars declared', () => {
    for (const size of [300, 777, 1000]) {
      const t = new TensorType([size], ScalarType.F32);
      const func = buildFunction(`q_edge_${size}`, [t, t], [t], (b, args) => {
        b.returnOp([b.add(args[0], args[1]).getResult(0)]);
      });
      const result = compile(func);
      expect(result.succeeded).toBe(true);
      const src = getSource(result, `q_edge_${size}`);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

// ────────────────────────────────────────────────────────────────────
// Epilogue fusion quality: matmul + unary ops
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — epilogue fusion unary ops', () => {
  it('matmul + exp: epilogue contains expf, not identity', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_exp', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.exp(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_exp');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/expf\(/);
  });

  it('matmul + tanh: epilogue contains tanhf', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_tanh', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.tanh(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_tanh');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/tanhf\(/);
  });

  it('matmul + sqrt: epilogue contains sqrtf', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_sqrt', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.sqrt(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_sqrt');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/sqrtf\(/);
  });

  it('matmul + abs: epilogue contains fabsf', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_abs', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.abs(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_abs');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/fabsf\(/);
  });

  it('matmul + log: epilogue contains logf', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_log', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.log(mm.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_log');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/logf\(/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Epilogue fusion quality: matmul + binary ops + activation
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — epilogue multi-op chains', () => {
  it('matmul + bias + relu: epilogue has fmaxf and add', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const bias = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_biasrelu', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_biasrelu');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/fmaxf\(/);
    expect(src).toMatch(/\+/);
  });

  it('matmul + bias: epilogue adds bias buffer', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const bias = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_bias', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.add(mm.getResult(0), args[2]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_bias');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
    const params = paramMatch[1].split(',').map(p => p.trim());
    expect(params.length).toBe(4);
  });

  it('matmul + scale: epilogue multiplies by scale buffer', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const scale = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_epi_scale', [lhs, rhs, scale], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.mul(mm.getResult(0), args[2]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_epi_scale');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
    expect(src).toMatch(/\*/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Epilogue: single kernel for matmul+epilogue
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — epilogue produces single kernel', () => {
  it('matmul + neg compiles to 1 kernel', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_1kern_neg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });

  it('matmul + bias + relu compiles to 1 kernel', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const bias = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_1kern_biasrelu', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      const biased = b.add(mm.getResult(0), args[2]);
      b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
    });
    expect(compile(func).listKernels()).toHaveLength(1);
  });
});

// ────────────────────────────────────────────────────────────────────
// Large reduction: blockIdx split
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — large reduction (blockIdx split)', () => {
  it('spatial > 256: uses blockIdx.x + threadIdx.x', () => {
    const tin = new TensorType([512, 64], ScalarType.F32);
    const tout = new TensorType([512], ScalarType.F32);
    const func = buildFunction('q_rsum_big', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rsum_big');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
  });

  it('spatial > 256: reduction loop is serial for loop', () => {
    const tin = new TensorType([1024, 32], ScalarType.F32);
    const tout = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('q_rsum_1k', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rsum_1k');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(countForLoops(src)).toBeGreaterThanOrEqual(1);
    expect(src).toMatch(/for.*32/);
  });

  it('spatial <= 256: only threadIdx, no blockIdx', () => {
    const tin = new TensorType([128, 64], ScalarType.F32);
    const tout = new TensorType([128], ScalarType.F32);
    const func = buildFunction('q_rsum_sm', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rsum_sm');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────
// 3D tensor: correct stride computation
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — 3D tensor stride correctness', () => {
  it('3D add: row-major strides in buffer access', () => {
    const t = new TensorType([4, 8, 16], ScalarType.F32);
    const func = buildFunction('q_3d_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_3d_add');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/\* 128/);
    expect(src).toMatch(/\* 16/);
    expect(countStoreStatements(src)).toBe(1);
  });

  it('3D add fuses to 0 for loops', () => {
    const t = new TensorType([4, 8, 16], ScalarType.F32);
    const func = buildFunction('q_3d_fuse', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_3d_fuse');
    expect(countForLoops(src)).toBe(0);
  });

  it('3D add: div and mod to recover original indices', () => {
    const t = new TensorType([4, 8, 16], ScalarType.F32);
    const func = buildFunction('q_3d_idx', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_3d_idx');
    expect(src).toMatch(/\/ 16/);
    expect(src).toMatch(/% 16/);
    expect(src).toMatch(/\/ 8/);
    expect(src).toMatch(/% 8/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Matmul structure: contraction loop
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — matmul contraction loop', () => {
  it('contraction dim becomes serial for loop', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_mm_contract', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_contract');
    expect(countForLoops(src)).toBeGreaterThanOrEqual(1);
    expect(src).toMatch(/for.*32/);
  });

  it('spatial dims use thread binding, not for loops', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_mm_spatial', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_spatial');
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────
// Large matmul: blockIdx splitting
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — large matmul tiling', () => {
  it('128x128 matmul: uses blockIdx for tiling', () => {
    const t = new TensorType([128, 128], ScalarType.F32);
    const func = buildFunction('q_mm_tile', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_tile');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'blockIdx.y')).toBe(true);
  });

  it('128x128 matmul: has guard conditions for non-divisible tiles', () => {
    const t = new TensorType([128, 128], ScalarType.F32);
    const func = buildFunction('q_mm_guard', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_guard');
    expect(src).toMatch(/if\s*\(/);
  });

  it('128x128 matmul: all vars declared', () => {
    const t = new TensorType([128, 128], ScalarType.F32);
    const func = buildFunction('q_mm_decl128', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_mm_decl128');
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

// ────────────────────────────────────────────────────────────────────
// Reduce + elementwise chain
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — reduce + elementwise chain', () => {
  it('sum + neg: negation applied to reduced result', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_sumneg', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const s = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      b.returnOp([b.neg(s.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const src = getSource(result, 'q_sumneg');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/-/);
  });

  it('sum + exp: exp applied to reduced result', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_sumexp', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const s = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      b.returnOp([b.exp(s.getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    const src = getSource(result, 'q_sumexp');
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

// ────────────────────────────────────────────────────────────────────
// Balanced braces/parens across all kernel types
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — balanced syntax across kernel types', () => {
  function checkBalanced(src) {
    expect((src.match(/\{/g) || []).length).toBe((src.match(/\}/g) || []).length);
    expect((src.match(/\(/g) || []).length).toBe((src.match(/\)/g) || []).length);
  }

  it('reduction kernel has balanced braces and parens', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('q_bal_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    checkBalanced(getSource(compile(func), 'q_bal_red'));
  });

  it('matmul kernel has balanced braces and parens', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_bal_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    checkBalanced(getSource(compile(func), 'q_bal_mm'));
  });

  it('matmul+epilogue kernel has balanced braces and parens', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_bal_epi', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    checkBalanced(getSource(compile(func), 'q_bal_epi'));
  });

  it('large matmul with guards has balanced braces', () => {
    const t = new TensorType([128, 128], ScalarType.F32);
    const func = buildFunction('q_bal_lmm', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    checkBalanced(getSource(compile(func), 'q_bal_lmm'));
  });
});

// ────────────────────────────────────────────────────────────────────
// No JS artifacts across all kernel types
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — no JS artifacts across kernel types', () => {
  function checkNoJS(src) {
    expect(src).not.toMatch(/Math\./);
    expect(src).not.toMatch(/\blet\s/);
    expect(src).not.toMatch(/\bfunction\s/);
    expect(src).not.toMatch(/=>/);
    expect(src).not.toContain('undefined');
    expect(src).not.toContain('NaN');
    expect(src).not.toContain('[object');
  }

  it('reduction kernel: no JS artifacts', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('q_js_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    checkNoJS(getSource(compile(func), 'q_js_red'));
  });

  it('matmul kernel: no JS artifacts', () => {
    const lhs = new TensorType([16, 32], ScalarType.F32);
    const rhs = new TensorType([32, 16], ScalarType.F32);
    const out = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_js_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    checkNoJS(getSource(compile(func), 'q_js_mm'));
  });

  it('matmul+neg kernel: no JS artifacts', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_js_mmneg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });
    checkNoJS(getSource(compile(func), 'q_js_mmneg'));
  });

  it('3D elementwise kernel: no JS artifacts', () => {
    const t = new TensorType([4, 8, 16], ScalarType.F32);
    const func = buildFunction('q_js_3d', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    checkNoJS(getSource(compile(func), 'q_js_3d'));
  });
});

// ────────────────────────────────────────────────────────────────────
// Kernel metadata correctness
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — kernel metadata', () => {
  it('1024-element add: gridDim*blockDim >= 1024', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('q_meta_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    const src = getSource(result, 'q_meta_add');
    const blockDimMatch = src.match(/\* (\d+)\)/);
    expect(result.succeeded).toBe(true);
    expect(result.listKernels()).toHaveLength(1);
  });

  it('matmul lists both input + output buffers as params', () => {
    const lhs = new TensorType([8, 16], ScalarType.F32);
    const rhs = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);
    const func = buildFunction('q_meta_mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_meta_mm');
    const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
    const params = paramMatch[1].split(',').map(p => p.trim());
    expect(params.length).toBe(3);
    for (const p of params) {
      expect(p).toMatch(/float\*/);
    }
  });

  it('reduction: 2 params (input + output)', () => {
    const tin = new TensorType([32, 64], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_meta_red', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_meta_red');
    const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
    const params = paramMatch[1].split(',').map(p => p.trim());
    expect(params.length).toBe(2);
  });
});

// ────────────────────────────────────────────────────────────────────
// Elementwise long chains
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — long elementwise chains', () => {
  it('add+sub+mul+neg: 4 ops fused into 1 store', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_long4', [t, t, t], [t], (b, args) => {
      const s = b.add(args[0], args[1]);
      const d = b.sub(s.getResult(0), args[2]);
      const p = b.mul(d.getResult(0), args[0]);
      b.returnOp([b.neg(p.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_long4');
    expect(countStoreStatements(src)).toBe(1);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).not.toMatch(/float\s+\w+\[\d+\]/);
  });

  it('exp(tanh(x)): nested math functions fused', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_nested_math', [t], [t], (b, args) => {
      const th = b.tanh(args[0]);
      b.returnOp([b.exp(th.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_nested_math');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/expf\(tanhf\(/);
  });

  it('sqrt(abs(x)): nested math functions fused', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('q_sqrt_abs', [t], [t], (b, args) => {
      const a = b.abs(args[0]);
      b.returnOp([b.sqrt(a.getResult(0)).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_sqrt_abs');
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/sqrtf\(fabsf\(/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Reduction: different combine ops
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — reduction combine ops', () => {
  it('max reduction: no undeclared vars, uses fmaxf', () => {
    const tin = new TensorType([64, 128], ScalarType.F32);
    const tout = new TensorType([64], ScalarType.F32);
    const func = buildFunction('q_rmax_big', [tin], [tout], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [1], 'max').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_rmax_big');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/fmaxf\(/);
    expect(src).toMatch(/(-INFINITY)/);
  });

  it('sum reduce axis 0: correct spatial/reduction split', () => {
    const tin = new TensorType([64, 32], ScalarType.F32);
    const tout = new TensorType([32], ScalarType.F32);
    const func = buildFunction('q_csum', [tin], [tout], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [0], 'sum').getResult(0)]);
    });
    const src = getSource(compile(func), 'q_csum');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
    expect(src).toMatch(/= 0f/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Multiple math ops: correct CUDA function names
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — math function mapping', () => {
  const ops = [
    { name: 'sin', fn: 'sinf', build: (b, x) => b.sin(x) },
    { name: 'cos', fn: 'cosf', build: (b, x) => b.cos(x) },
    { name: 'floor', fn: 'floorf', build: (b, x) => b.floor(x) },
    { name: 'ceil', fn: 'ceilf', build: (b, x) => b.ceil(x) },
  ];

  for (const { name, fn, build } of ops) {
    it(`${name} maps to ${fn} for f32`, () => {
      const t = new TensorType([64], ScalarType.F32);
      const func = buildFunction(`q_math_${name}`, [t], [t], (b, args) => {
        b.returnOp([build(b, args[0]).getResult(0)]);
      });
      const src = getSource(compile(func), `q_math_${name}`);
      expect(src).toMatch(new RegExp(`${fn}\\(`));
      expect(findUndeclaredVars(src)).toEqual([]);
    });
  }
});

// ────────────────────────────────────────────────────────────────────
// Non-divisible split: guard conditions
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — non-divisible split guards', () => {
  it('300 elements (not divisible by 256): has bounds guard or valid code', () => {
    const t = new TensorType([300], ScalarType.F32);
    const func = buildFunction('q_guard_300', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_guard_300');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasThreadBinding(src, 'blockIdx.x')).toBe(true);
    expect(hasThreadBinding(src, 'threadIdx.x')).toBe(true);
    expect(src).toMatch(/if\s*\(/);
  });

  it('1000 elements: guard prevents out-of-bounds', () => {
    const t = new TensorType([1000], ScalarType.F32);
    const func = buildFunction('q_guard_1000', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_guard_1000');
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/if\s*\(/);
    expect(src).toMatch(/< 1000/);
  });
});

// ────────────────────────────────────────────────────────────────────
// 2D elementwise: correct index decomposition
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — 2D index decomposition', () => {
  it('32x64 add: decomposes fused index with / and %', () => {
    const t = new TensorType([32, 64], ScalarType.F32);
    const func = buildFunction('q_2d_decomp', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_2d_decomp');
    expect(src).toMatch(/\/ 64/);
    expect(src).toMatch(/% 64/);
    expect(countForLoops(src)).toBe(0);
    expect(countStoreStatements(src)).toBe(1);
  });

  it('16x16 neg: single store with 2D indexing', () => {
    const t = new TensorType([16, 16], ScalarType.F32);
    const func = buildFunction('q_2d_neg', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });
    const src = getSource(compile(func), 'q_2d_neg');
    expect(countStoreStatements(src)).toBe(1);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/-/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Matmul sizes: various configurations
// ────────────────────────────────────────────────────────────────────

describe('GPU kernel quality — matmul size variants', () => {
  const configs = [
    { name: '4x8_8x4', M: 4, K: 8, N: 4 },
    { name: '32x32_32x32', M: 32, K: 32, N: 32 },
    { name: '16x64_64x16', M: 16, K: 64, N: 16 },
  ];

  for (const { name, M, K, N } of configs) {
    it(`${name}: all vars declared, no noop, valid CUDA`, () => {
      const lhs = new TensorType([M, K], ScalarType.F32);
      const rhs = new TensorType([K, N], ScalarType.F32);
      const out = new TensorType([M, N], ScalarType.F32);
      const func = buildFunction(`q_mm_${name}`, [lhs, rhs], [out], (b, args) => {
        b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
      });
      const src = getSource(compile(func), `q_mm_${name}`);
      expect(findUndeclaredVars(src)).toEqual([]);
      expect(hasNoopStore(src)).toBe(false);
      expect(src).toMatch(/__global__/);
      expect(src).toMatch(/= 0f/);
    });
  }
});

// ────────────────────────────────────────────────────────────────────
// Scheduled kernel: __shared__ promotion + __syncthreads (TVM/XLA fix)
// ────────────────────────────────────────────────────────────────────

function compileGPU(model, inputs, opts) {
  return modelCompile(model, inputs, { target: GPUTarget(), ...opts });
}

describe('GPU kernel quality — scheduled shared memory + barriers', () => {
  it('scheduled multi-stage kernel uses __shared__ and __syncthreads', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('__shared__');
    expect(src).toContain('__syncthreads()');
  });

  it('scheduled kernel emits bounds guard for smaller stages', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toMatch(/if \(threadIdx\.x < 2\)/);
  });

  it('scheduled kernel has balanced braces', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('unscheduled kernel uses local array, not __shared__', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x]);
    const src = compiled.source();

    expect(src).not.toContain('__syncthreads()');
  });

  it('scheduled single-layer has no barriers or __shared__ promotion', () => {
    const model = new Sequential(new Linear(4, 4));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).not.toContain('__syncthreads()');
  });

  it('wide bottleneck: Linear(4,64)->ReLU->Linear(64,2) scheduled', () => {
    const model = new Sequential(new Linear(4, 64), new ReLU(), new Linear(64, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('__shared__');
    expect(src).toContain('__syncthreads()');
    expect(src).toMatch(/if \(threadIdx\.x < 2\)/);
  });

  it('5-layer deep MLP scheduled compiles with valid CUDA', () => {
    const model = new Sequential(
      new Linear(4, 16), new ReLU(),
      new Linear(16, 32), new ReLU(),
      new Linear(32, 16), new ReLU(),
      new Linear(16, 8), new ReLU(),
      new Linear(8, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
    expect(src).toContain('__shared__');
    expect(src).toContain('__syncthreads()');
  });

  it('multi-layer ReLU scheduled', () => {
    const model = new Sequential(
      new Linear(4, 8), new ReLU(),
      new Linear(8, 8), new ReLU(),
      new Linear(8, 8), new ReLU(),
      new Linear(8, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('__shared__');
    expect(src).toContain('__syncthreads()');
    expect(src).toContain('fmaxf(');

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('same-extent model scheduled has no bounds guards', () => {
    const model = new Sequential(
      new Linear(4, 8), new ReLU(),
      new Linear(8, 8), new ReLU(),
      new Linear(8, 8),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).not.toMatch(/if \(threadIdx\.\w < \d+\)/);
  });

  it('contracting model: Linear(32,16)->ReLU->Linear(16,4)->ReLU->Linear(4,1)', () => {
    const model = new Sequential(
      new Linear(32, 16), new ReLU(),
      new Linear(16, 4), new ReLU(),
      new Linear(4, 1),
    );
    const x = tensor([Array.from({ length: 32 }, (_, i) => i)]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('__shared__');
    expect(src).toContain('__syncthreads()');
    expect(src).toMatch(/if \(threadIdx\.x < 4\)/);
    expect(src).toMatch(/if \(threadIdx\.x < 1\)/);

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('no private vars when shared promotion active', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const sharedVars = [...src.matchAll(/__shared__\s+\w+\s+(buf_\d+)/g)].map(m => m[1]);
    const localVars = [...src.matchAll(/^\s+float (buf_\d+)\[\d+\]/gm)].map(m => m[1]);

    expect(sharedVars.length).toBeGreaterThan(0);
    expect(localVars.length).toBe(0);
  });
});
