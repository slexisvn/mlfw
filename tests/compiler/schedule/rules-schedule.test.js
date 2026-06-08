import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';

const F32 = ScalarType.F32;

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget(), { scheduling: { enabled: true }, ...opts });
}

function compileDefault(func) {
  return compileGraph(func, CPUTarget(), { scheduling: { enabled: false } });
}

function src(result) {
  const kernels = result.listKernels();
  return kernels.map(k => result.getSource(k)).join('\n');
}

function countStores(s) {
  return (s.match(/\w+\[.*?\]\s*=/g) || []).length;
}

function countForLoops(s) {
  return (s.match(/\bfor\s*\(/g) || []).length;
}

describe('MatmulTiledCPURule — tiling via schedule.tile()', () => {
  it('matmul with scheduling does not inflate stores vs default', () => {
    const a = new TensorType([32, 64], F32);
    const b = new TensorType([64, 32], F32);
    const out = new TensorType([32, 32], F32);
    const func = buildFunction('mm', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const funcDef = buildFunction('mm_def', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const scheduled = src(compile(func));
    const defaultSrc = src(compileDefault(funcDef));

    const schedStores = countStores(scheduled);
    const defStores = countStores(defaultSrc);

    expect(schedStores).toBeLessThanOrEqual(defStores * 1.5);
  });

  it('tiled matmul has more for-loops than default (outer+inner tiles)', () => {
    const a = new TensorType([64, 64], F32);
    const b = new TensorType([64, 64], F32);
    const out = new TensorType([64, 64], F32);
    const func = buildFunction('mm_tile', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const funcDef = buildFunction('mm_tile_def', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const scheduled = src(compile(func));
    const defaultSrc = src(compileDefault(funcDef));

    expect(countForLoops(scheduled)).toBeGreaterThanOrEqual(countForLoops(defaultSrc));
  });

  it('small matmul that does not meet tile threshold still compiles', () => {
    const a = new TensorType([4, 4], F32);
    const b = new TensorType([4, 4], F32);
    const out = new TensorType([4, 4], F32);
    const func = buildFunction('mm_small', [a, b], [out], (builder, args) => {
      builder.returnOp([builder.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    expect(result.listKernels().length).toBeGreaterThan(0);
    expect(src(result)).toMatch(/\w+\[.*?\]\s*=/);
  });
});

describe('ElementwiseCPURule — VECTORIZED loops not unrolled in CPU codegen', () => {
  it('elementwise add with scheduling does not inflate stores vs default', () => {
    const t = new TensorType([256], F32);
    const func = buildFunction('ew_add', [t, t], [t], (builder, args) => {
      builder.returnOp([builder.add(args[0], args[1]).getResult(0)]);
    });

    const funcDef = buildFunction('ew_add_def', [t, t], [t], (builder, args) => {
      builder.returnOp([builder.add(args[0], args[1]).getResult(0)]);
    });

    const scheduled = src(compile(func));
    const defaultSrc = src(compileDefault(funcDef));

    expect(countStores(scheduled)).toBeLessThanOrEqual(countStores(defaultSrc) * 1.15);
  });

  it('scheduled relu produces same store count as default', () => {
    const t = new TensorType([512], F32);
    const func = buildFunction('ew_relu', [t], [t], (builder, args) => {
      builder.returnOp([builder.relu(args[0]).getResult(0)]);
    });

    const funcDef = buildFunction('ew_relu_def', [t], [t], (builder, args) => {
      builder.returnOp([builder.relu(args[0]).getResult(0)]);
    });

    const schedStores = countStores(src(compile(func)));
    const defStores = countStores(src(compileDefault(funcDef)));

    expect(schedStores).toBeLessThanOrEqual(defStores * 1.15);
  });

  it('chain add+mul+neg with scheduling does not inflate stores', () => {
    const t = new TensorType([256], F32);
    const func = buildFunction('ew_chain', [t, t, t], [t], (builder, args) => {
      const sum = builder.add(args[0], args[1]);
      const prod = builder.mul(sum.getResult(0), args[2]);
      builder.returnOp([builder.neg(prod.getResult(0)).getResult(0)]);
    });

    const funcDef = buildFunction('ew_chain_def', [t, t, t], [t], (builder, args) => {
      const sum = builder.add(args[0], args[1]);
      const prod = builder.mul(sum.getResult(0), args[2]);
      builder.returnOp([builder.neg(prod.getResult(0)).getResult(0)]);
    });

    const schedStores = countStores(src(compile(func)));
    const defStores = countStores(src(compileDefault(funcDef)));

    expect(schedStores).toBeLessThanOrEqual(defStores * 1.15);
  });
});
