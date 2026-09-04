import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { DecompositionPass } from '../../../../src/compiler/passes/decompose/decomposition_pass.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';

function run(func) {
  return new DecompositionPass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('attention decomposition reaches a fixed point', () => {
  it.each([false, true])('fully lowers attention including its new softmax and computes causal=%s correctly', (causal) => {
    const qType = new TensorType([1, 2, 2], 'f32');
    const kvType = new TensorType([1, 3, 2], 'f32');
    const func = buildFunction('attention', [qType, kvType, kvType], [qType], (b, [q, k, v]) => {
      b.returnOp(b.scaledDotProductAttention(q, k, v, 0.5, causal).results);
    });
    expect(run(func)).toBe(PassResult.CHANGED);
    expect(func.findOp((op) => op.opName === 'scaled_dot_product_attention' || op.opName === 'softmax')).toBeNull();
    expect(run(func)).toBe(PassResult.UNCHANGED);
    const q = new Float32Array([1, 0, 0, 1]);
    const k = new Float32Array([2, 0, 0, 2, 1, 1]);
    const v = new Float32Array([1, 2, 3, 4, 10, 20]);
    const expected = [];
    for (let row = 0; row < 2; row++) {
      const scores = [];
      for (let col = 0; col < 3; col++) {
        const allowed = !causal || col <= row + 1;
        scores.push(allowed ? Math.exp(0.5 * (q[row * 2] * k[col * 2] + q[row * 2 + 1] * k[col * 2 + 1])) : 0);
      }
      const total = scores.reduce((sum, x) => sum + x, 0);
      for (let dim = 0; dim < 2; dim++) expected.push(scores.reduce((sum, score, col) => sum + score * v[col * 2 + dim], 0) / total);
    }
    const output = new Float32Array(4);
    compileGraph(func, CPUTarget()).run('attention', q, k, v, output);
    for (let i = 0; i < output.length; i++) expect(output[i]).toBeCloseTo(expected[i], 5);
  });
});

describe('layer_norm decomposition', () => {
  it('computes mean from input (has reduce ops), unlike batch_norm', () => {
    const inputT = new TensorType([4, 8], ScalarType.F32);
    const paramT = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [inputT, paramT, paramT], [inputT], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2], 1, 1e-5).getResult(0)]);
    });

    run(func);

    const reduces = func.findOps(op => op.opName === 'reduce');
    expect(reduces.length).toBeGreaterThanOrEqual(2);

    const meanReduce = reduces.find(r => r.getAttr('reduce_type') === 'mean');
    expect(meanReduce).toBeDefined();
    expect(meanReduce.getOperand(0)).toBe(func.args[0]);
  });

  it('epsilon value from attr propagates into constant, not a default', () => {
    const inputT = new TensorType([4, 8], ScalarType.F32);
    const paramT = new TensorType([8], ScalarType.F32);

    for (const eps of [1e-5, 1e-6, 1e-3]) {
      const func = buildFunction('f', [inputT, paramT, paramT], [inputT], (b, args) => {
        b.returnOp([b.layernorm(args[0], args[1], args[2], 1, eps).getResult(0)]);
      });

      run(func);

      const constants = func.findOps(op => op.opName === 'constant');
      const epsConst = constants.find(c => c.getAttr('value') === eps);
      expect(epsConst).toBeDefined();
    }
  });

  it('reduce axis changes with different normalization dimensions', () => {
    const shapes = [
      { input: [4, 8], param: [4], axis: 0, expectedReduceShape: [8] },
      { input: [4, 8], param: [8], axis: 1, expectedReduceShape: [4] },
      { input: [2, 4, 8], param: [8], axis: 2, expectedReduceShape: [2, 4] },
    ];

    for (const { input, param, axis, expectedReduceShape } of shapes) {
      const func = buildFunction('f',
        [new TensorType(input, ScalarType.F32), new TensorType(param, ScalarType.F32), new TensorType(param, ScalarType.F32)],
        [new TensorType(input, ScalarType.F32)],
        (b, args) => {
          b.returnOp([b.layernorm(args[0], args[1], args[2], axis, 1e-5).getResult(0)]);
        });

      run(func);

      const reduces = func.findOps(op => op.opName === 'reduce');
      for (const r of reduces) {
        expect(r.getAttr('dimensions')).toEqual([axis]);
        expect(r.getResult(0).type.shape).toEqual(expectedReduceShape);
      }

      expect(retVal(func).type.shape).toEqual(input);
    }
  });

  it('gamma and beta connect to the final scale and shift ops', () => {
    const inputT = new TensorType([4, 8], ScalarType.F32);
    const paramT = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [inputT, paramT, paramT], [inputT], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2], 1, 1e-5).getResult(0)]);
    });

    run(func);

    const broadcasts = func.findOps(op => op.opName === 'broadcast_in_dim');
    const gammaBcast = broadcasts.find(bc => bc.getOperand(0) === func.args[1]);
    const betaBcast = broadcasts.find(bc => bc.getOperand(0) === func.args[2]);
    expect(gammaBcast).toBeDefined();
    expect(betaBcast).toBeDefined();
    expect(gammaBcast.getResult(0).type.shape).toEqual([4, 8]);
    expect(betaBcast.getResult(0).type.shape).toEqual([4, 8]);
  });
});

describe('batch_norm decomposition', () => {
  it('uses provided mean/variance directly, never computes reduce', () => {
    const inputT = new TensorType([2, 3, 4, 4], ScalarType.F32);
    const paramT = new TensorType([3], ScalarType.F32);
    const func = buildFunction('f', [inputT, paramT, paramT, paramT, paramT], [inputT], (b, args) => {
      b.returnOp([b.batchnorm(args[0], args[1], args[2], args[3], args[4], 1, 1e-5).getResult(0)]);
    });

    run(func);

    const reduces = func.findOps(op => op.opName === 'reduce');
    expect(reduces).toHaveLength(0);

    const subOp = func.findOp(op => op.opName === 'sub');
    const meanBcast = subOp.getOperand(1).definingOp;
    expect(meanBcast.getOperand(0)).toBe(func.args[3]);
  });

  it('broadcasts params along channel axis, not spatial axes', () => {
    const inputT = new TensorType([2, 16, 32, 32], ScalarType.F32);
    const paramT = new TensorType([16], ScalarType.F32);
    const func = buildFunction('f', [inputT, paramT, paramT, paramT, paramT], [inputT], (b, args) => {
      b.returnOp([b.batchnorm(args[0], args[1], args[2], args[3], args[4], 1, 1e-5).getResult(0)]);
    });

    run(func);

    const broadcasts = func.findOps(op => op.opName === 'broadcast_in_dim');
    const paramBcasts = broadcasts.filter(bc => {
      const src = bc.getOperand(0);
      return src === func.args[1] || src === func.args[2] ||
             src === func.args[3] || src === func.args[4] ||
             (src.definingOp && src.type.shape.length < 4);
    });

    for (const bc of paramBcasts) {
      if (bc.getResult(0).type.shape.length === 4) {
        expect(bc.getResult(0).type.shape).toEqual([2, 16, 32, 32]);
      }
    }

    expect(retVal(func).type.shape).toEqual([2, 16, 32, 32]);
  });

  it('rsqrt receives variance+epsilon, not raw variance', () => {
    const inputT = new TensorType([2, 3, 4, 4], ScalarType.F32);
    const paramT = new TensorType([3], ScalarType.F32);
    const eps = 1e-5;
    const func = buildFunction('f', [inputT, paramT, paramT, paramT, paramT], [inputT], (b, args) => {
      b.returnOp([b.batchnorm(args[0], args[1], args[2], args[3], args[4], 1, eps).getResult(0)]);
    });

    run(func);

    const rsqrt = func.findOp(op => op.opName === 'rsqrt');
    const addOp = rsqrt.getOperand(0).definingOp;
    expect(addOp.opName).toBe('add');
    expect(addOp.getOperand(0)).toBe(func.args[4]);
  });
});
