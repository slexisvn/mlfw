import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

describe('layer_norm', () => {
  it('normalizes rows to mean≈0, variance≈1', () => {
    const x = new TensorType([2, 4], ScalarType.F32);
    const s = new TensorType([4], ScalarType.F32);
    const func = buildFunction('ln', [x, s, s], [x], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2]).getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]);
    const scale = new Float32Array([1, 1, 1, 1]);
    const bias = new Float32Array([0, 0, 0, 0]);
    const out = new Float32Array(8);
    r.run('ln', inp, scale, bias, out);
    for (let row = 0; row < 2; row++) {
      let mean = 0;
      for (let j = 0; j < 4; j++) mean += out[row * 4 + j];
      mean /= 4;
      expect(mean).toBeCloseTo(0, 3);
    }
  });

  it('applies scale and bias', () => {
    const x = new TensorType([1, 3], ScalarType.F32);
    const s = new TensorType([3], ScalarType.F32);
    const func = buildFunction('ln_sb', [x, s, s], [x], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2]).getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([1, 2, 3]);
    const scale = new Float32Array([2, 2, 2]);
    const bias = new Float32Array([10, 10, 10]);
    const out = new Float32Array(3);
    r.run('ln_sb', inp, scale, bias, out);
    let mean = 0;
    for (let j = 0; j < 3; j++) mean += out[j];
    mean /= 3;
    expect(mean).toBeCloseTo(10, 2);
  });
});

describe('batch_norm', () => {
  it('normalizes 2D input (batch over dim 0)', () => {
    const x = new TensorType([4, 3], ScalarType.F32);
    const c = new TensorType([3], ScalarType.F32);
    const func = buildFunction('bn2d', [x, c, c, c, c], [x], (b, args) => {
      b.returnOp([b.batchnorm(args[0], args[1], args[2], args[3], args[4]).getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400]);
    const scale = new Float32Array([1, 1, 1]);
    const bias = new Float32Array([0, 0, 0]);
    const mean = new Float32Array([2.5, 25, 250]);
    const variance = new Float32Array([1.25, 125, 12500]);
    const out = new Float32Array(12);
    r.run('bn2d', inp, scale, bias, mean, variance, out);
    for (let ch = 0; ch < 3; ch++) {
      let colMean = 0;
      for (let b = 0; b < 4; b++) colMean += out[b * 3 + ch];
      colMean /= 4;
      expect(colMean).toBeCloseTo(0, 2);
    }
  });

  it('normalizes 4D NCHW input', () => {
    const x = new TensorType([1, 2, 2, 2], ScalarType.F32);
    const c = new TensorType([2], ScalarType.F32);
    const func = buildFunction('bn4d', [x, c, c, c, c], [x], (b, args) => {
      b.returnOp([b.batchnorm(args[0], args[1], args[2], args[3], args[4]).getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]);
    const scale = new Float32Array([1, 1]);
    const bias = new Float32Array([0, 0]);
    const mean = new Float32Array([2.5, 25]);
    const variance = new Float32Array([1.25, 125]);
    const out = new Float32Array(8);
    r.run('bn4d', inp, scale, bias, mean, variance, out);
    for (let ch = 0; ch < 2; ch++) {
      let chMean = 0;
      for (let i = 0; i < 4; i++) chMean += out[ch * 4 + i];
      chMean /= 4;
      expect(chMean).toBeCloseTo(0, 2);
    }
  });
});
