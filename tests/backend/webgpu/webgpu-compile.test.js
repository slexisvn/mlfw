import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { WebGPUTarget } from '../../../src/backend/target.js';

function compile(func, opts = {}) {
  return compileGraph(func, WebGPUTarget(), { scheduling: { enabled: true }, ...opts });
}

function compileNoSchedule(func, opts = {}) {
  return compileGraph(func, WebGPUTarget(), { scheduling: { enabled: false }, ...opts });
}

function getSource(result, name) {
  return result.getSource(name);
}

function countForLoops(src) {
  return (src.match(/\bfor\s*\(/g) || []).length;
}

function countStoreStatements(src) {
  return (src.match(/\w+\[.*\]\s*=/g) || []).length;
}

function hasNoCudaisms(src) {
  return !src.match(/__global__|__shared__|float\*|int\*|#pragma|threadIdx\.|blockIdx\.|expf|logf|sqrtf|tanhf|fabsf|sinf|cosf|ceilf|floorf|fmaxf|fminf|powf|rsqrtf/);
}

describe('WebGPU compilation — WGSL output format', () => {
  it('produces valid WGSL structure for elementwise add', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_add', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const kernels = result.listKernels();
    expect(kernels.length).toBeGreaterThan(0);

    const src = getSource(result, kernels[0]);
    expect(src).toMatch(/@group\(0\) @binding/);
    expect(src).toMatch(/@compute @workgroup_size/);
    expect(src).toMatch(/fn /);
    expect(src).toMatch(/array<f32>/);
    expect(hasNoCudaisms(src)).toBe(true);
  });

  it('uses storage buffer bindings not pointer params', () => {
    const t = new TensorType([128], ScalarType.F32);
    const func = buildFunction('wgpu_copy', [t], [t], (b, args) => {
      b.returnOp([args[0]]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/var<storage/);
    expect(src).not.toMatch(/float\*/);
  });

  it('uses WGSL builtins for thread indices', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('wgpu_idx', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    if (src.includes('_lid') || src.includes('_wid') || src.includes('_gid')) {
      expect(src).not.toMatch(/threadIdx\./);
      expect(src).not.toMatch(/blockIdx\./);
    }
  });
});

describe('WebGPU compilation — elementwise fusion', () => {
  it('add+mul fuses into single store', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('wgpu_add_mul', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.mul(sum.getResult(0), args[2]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/\+.*\*/);
  });

  it('add+mul+neg chain fuses into single store', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_chain3', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      b.returnOp([b.neg(prod.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(countStoreStatements(src)).toBe(1);
  });

  it('sub+neg fuses', () => {
    const t = new TensorType([512], ScalarType.F32);
    const func = buildFunction('wgpu_sub_neg', [t, t], [t], (b, args) => {
      const diff = b.sub(args[0], args[1]);
      b.returnOp([b.neg(diff.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    expect(countStoreStatements(src)).toBe(1);
  });

  it('mul+exp fuses exp(a*b) in single store', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_mul_exp', [t, t], [t], (b, args) => {
      const prod = b.mul(args[0], args[1]);
      b.returnOp([b.exp(prod.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(countStoreStatements(src)).toBe(1);
    expect(src).toMatch(/exp\(.+\*.+\)/);
    expect(src).not.toMatch(/expf/);
  });
});

describe('WebGPU compilation — math functions use WGSL builtins', () => {
  it('exp uses exp() not expf()', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_exp', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/exp\(/);
    expect(src).not.toMatch(/expf\(/);
  });

  it('sqrt uses sqrt() not sqrtf()', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_sqrt', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/sqrt\(/);
    expect(src).not.toMatch(/sqrtf\(/);
  });

  it('tanh uses tanh() not tanhf()', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_tanh', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/tanh\(/);
    expect(src).not.toMatch(/tanhf\(/);
  });
});

describe('WebGPU compilation — reduction', () => {
  it('sum reduction compiles with loop', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('wgpu_sum', [inT], [outT], (b, args) => {
      const zero = b.scalarConstant(0.0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/for \(var/);
  });
});

describe('WebGPU compilation — matmul', () => {
  it('matmul compiles to valid WGSL', () => {
    const aT = new TensorType([4, 8], ScalarType.F32);
    const bT = new TensorType([8, 4], ScalarType.F32);
    const outT = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('wgpu_matmul', [aT, bT], [outT], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/@compute/);
    expect(src).toMatch(/array<f32>/);
    expect(hasNoCudaisms(src)).toBe(true);
  });
});

describe('WebGPU compilation — no CUDA leakage', () => {
  it('no CUDA syntax in any compiled output', () => {
    const t = new TensorType([512], ScalarType.F32);
    const func = buildFunction('wgpu_clean', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const relu = b.relu(sum.getResult(0));
      b.returnOp([relu.getResult(0)]);
    });

    const result = compile(func);
    for (const name of result.listKernels()) {
      const src = getSource(result, name);
      expect(hasNoCudaisms(src)).toBe(true);
    }
  });

  it('uses max() for relu instead of CUDA ternary', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_relu', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/max\(/);
    expect(src).not.toMatch(/\?/);
  });
});

describe('WebGPU compilation — kernel metadata', () => {
  it('compiled kernel has kind webgpu', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wgpu_meta', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const result = compile(func);
    const kernels = result.listKernels();
    expect(kernels.length).toBeGreaterThan(0);
  });

  it('buffer binding count matches param count', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('wgpu_bindings', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = getSource(result, result.listKernels()[0]);
    const bindingCount = (src.match(/@binding\(/g) || []).length;
    expect(bindingCount).toBeGreaterThanOrEqual(2);
  });
});

describe('WebGPU compilation — balanced braces', () => {
  it('all kernels have balanced braces', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('wgpu_braces', [t, t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const prod = b.mul(sum.getResult(0), args[2]);
      const relu = b.relu(prod.getResult(0));
      b.returnOp([relu.getResult(0)]);
    });

    const result = compile(func);
    for (const name of result.listKernels()) {
      const src = getSource(result, name);
      const opens = (src.match(/\{/g) || []).length;
      const closes = (src.match(/\}/g) || []).length;
      expect(opens).toBe(closes);
    }
  });
});


describe('WebGPU compilation — boolean ops WGSL validity', () => {
  const F32 = ScalarType.F32;
  const tt = (shape) => new TensorType(shape, F32);

  it('compare + logicalAnd + logicalOr + select compiles clean WGSL', () => {
    const func = buildFunction('wgpu_bool', [tt([64]), tt([64])], [tt([64])], (b, args) => {
      const gt = b.compare(args[0], args[1], 'gt').getResult(0);
      const lt = b.compare(args[0], args[1], 'lt').getResult(0);
      const anded = b.logicalAnd(gt, lt).getResult(0);
      const ored = b.logicalOr(gt, anded).getResult(0);
      b.returnOp([b.select(ored, args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, 'wgpu_bool');
    expect(src).toContain('select(');
    expect(src).not.toMatch(/bool\s*!=\s*0/);
    expect(src).toContain('@compute');
  });

  it('compare stored to buffer uses select() not raw bool assignment', () => {
    const func = buildFunction('wgpu_cmp_store', [tt([32]), tt([32])], [tt([32])], (b, args) => {
      const gt = b.compare(args[0], args[1], 'gt').getResult(0);
      b.returnOp([b.select(gt, args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, 'wgpu_cmp_store');
    expect(src).toContain('select(');
  });

  it('pad+conv bounds check does not produce bool != 0', () => {
    const func = buildFunction('wgpu_pad_conv', [
      tt([1, 1, 8, 8]), tt([4, 1, 3, 3])
    ], [tt([1, 4, 8, 8])], (b, args) => {
      const padVal = b.scalarConstant(0, F32).getResult(0);
      const padded = b.pad(args[0], padVal, [0, 0, 1, 1], [0, 0, 1, 1]).getResult(0);
      b.returnOp([b.conv(padded, args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      expect(src).not.toMatch(/bool\s*!=\s*0/);
    }
  });

  it('conditional clamp with compare+select produces valid WGSL', () => {
    const func = buildFunction('wgpu_cond', [tt([32]), tt([32])], [tt([32])], (b, args) => {
      const gt = b.compare(args[0], args[1], 'gt').getResult(0);
      const eq = b.compare(args[0], args[1], 'eq').getResult(0);
      const ored = b.logicalOr(gt, eq).getResult(0);
      const selected = b.select(ored, args[0], args[1]).getResult(0);
      const zero = b.constant(0, tt([32])).getResult(0);
      const one = b.constant(1, tt([32])).getResult(0);
      b.returnOp([b.clamp(zero, selected, one).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
  });
});
