import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/backward_builder.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import '../../../src/compiler/ad/index.js';
import { F32 } from '../../_utils/ir_fixture.js';


function t(shape) {
  return new TensorType(shape, F32);
}

function numel(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function buildAndCompileBackward(forwardFunc) {
  const bwdBuilder = new BackwardGraphBuilder();
  const result = bwdBuilder.build(forwardFunc);
  const bwdFunc = result.backwardFunc;
  const compiled = compileGraph(bwdFunc, CPUTarget());
  const funcName = compiled.listKernels()[0];

  return {
    run(...inputBuffers) {
      const outputBuffers = bwdFunc.outputTypes.map(
        outType => new Float32Array(Math.max(numel(outType.shape), 1))
      );
      compiled.run(funcName, ...inputBuffers, ...outputBuffers);
      return outputBuffers;
    },
    inputTypes: bwdFunc.inputTypes,
    outputTypes: bwdFunc.outputTypes,
    savedValues: result.savedValues,
    bwdFunc,
  };
}

describe('Compiled backward numerical correctness', () => {
  it('add backward: grad passthrough', () => {
    const func = buildFunction('add_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const numInputs = bwd.inputTypes.length;

    const inputs = [];
    for (let i = 0; i < numInputs; i++) {
      const shape = bwd.inputTypes[i].shape;
      inputs.push(new Float32Array(numel(shape)));
    }
    inputs[0].set([1, 2, 3, 4]);

    const [gradA, gradB] = bwd.run(...inputs);

    expect([...gradA]).toEqual([1, 2, 3, 4]);
    expect([...gradB]).toEqual([1, 2, 3, 4]);
  });

  it('mul backward: grad * other', () => {
    const func = buildFunction('mul_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const numInputs = bwd.inputTypes.length;

    const a = [2, 3, 4, 5];
    const b_val = [6, 7, 8, 9];

    const inputs = [];
    for (let i = 0; i < numInputs; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1, 1]);
    inputs[1].set(a);
    inputs[2].set(b_val);

    const [gradA, gradB] = bwd.run(...inputs);

    for (let i = 0; i < 4; i++) {
      expect(gradA[i]).toBeCloseTo(b_val[i], 4);
      expect(gradB[i]).toBeCloseTo(a[i], 4);
    }
  });

  it('neg backward: negates gradient', () => {
    const func = buildFunction('neg_fwd', [t([4])], [t([4])], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const numInputs = bwd.inputTypes.length;

    const inputs = [];
    for (let i = 0; i < numInputs; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, -2, 3, -4]);

    const [gradX] = bwd.run(...inputs);

    expect([...gradX]).toEqual([-1, 2, -3, 4]);
  });

  it('exp backward: grad * exp(x)', () => {
    const func = buildFunction('exp_fwd', [t([4])], [t([4])], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const numInputs = bwd.inputTypes.length;

    const x = [0, 1, -1, 2];
    const expX = x.map(v => Math.exp(v));

    const inputs = [];
    for (let i = 0; i < numInputs; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1, 1]);

    let savedIdx = 1;
    for (let i = 1; i < numInputs; i++) {
      if (numel(bwd.inputTypes[i].shape) === 4) {
        if (savedIdx === 1) {
          inputs[i].set(x);
          savedIdx++;
        } else {
          inputs[i].set(expX);
        }
      }
    }

    const [gradX] = bwd.run(...inputs);

    for (let i = 0; i < 4; i++) {
      expect(gradX[i]).toBeCloseTo(expX[i], 4);
    }
  });

  it('sigmoid backward: grad * sig * (1 - sig)', () => {
    const func = buildFunction('sig_fwd', [t([3])], [t([3])], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const numInputs = bwd.inputTypes.length;

    const x = [0, 1, -1];
    const sig = x.map(v => 1 / (1 + Math.exp(-v)));
    const expectedGrad = sig.map(s => s * (1 - s));

    const inputs = [];
    for (let i = 0; i < numInputs; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1]);

    let savedIdx = 1;
    for (let i = 1; i < numInputs; i++) {
      if (numel(bwd.inputTypes[i].shape) === 3) {
        if (savedIdx === 1) {
          inputs[i].set(x);
          savedIdx++;
        } else {
          inputs[i].set(sig);
        }
      }
    }

    const [gradX] = bwd.run(...inputs);

    for (let i = 0; i < 3; i++) {
      expect(gradX[i]).toBeCloseTo(expectedGrad[i], 4);
    }
  });

  it('square backward: grad * 2x', () => {
    const func = buildFunction('square_fwd', [t([4])], [t([4])], (b, args) => {
      b.returnOp([b.square(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const x = [1, 2, -3, 4];
    const inputs = [];
    for (let i = 0; i < bwd.inputTypes.length; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1, 1]);
    for (let i = 1; i < inputs.length; i++) inputs[i].set(x);

    const [gradX] = bwd.run(...inputs);
    for (let i = 0; i < 4; i++) {
      expect(gradX[i]).toBeCloseTo(2 * x[i], 4);
    }
  });

  it('reciprocal backward: grad * -1/x²', () => {
    const func = buildFunction('recip_fwd', [t([4])], [t([4])], (b, args) => {
      b.returnOp([b.reciprocal(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const x = [1, 2, 0.5, 4];
    const recipX = x.map(v => 1 / v);

    const inputs = [];
    for (let i = 0; i < bwd.inputTypes.length; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1, 1]);

    let savedIdx = 1;
    for (let i = 1; i < inputs.length; i++) {
      if (numel(bwd.inputTypes[i].shape) === 4) {
        if (savedIdx === 1) {
          inputs[i].set(x);
          savedIdx++;
        } else {
          inputs[i].set(recipX);
        }
      }
    }

    const [gradX] = bwd.run(...inputs);
    for (let i = 0; i < 4; i++) {
      expect(gradX[i]).toBeCloseTo(-1 / (x[i] * x[i]), 3);
    }
  });

  it('log2 backward: grad / (x * ln2)', () => {
    const func = buildFunction('log2_fwd', [t([4])], [t([4])], (b, args) => {
      b.returnOp([b.log2(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const x = [1, 2, 4, 0.5];
    const inputs = [];
    for (let i = 0; i < bwd.inputTypes.length; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1, 1]);
    for (let i = 1; i < inputs.length; i++) {
      if (numel(bwd.inputTypes[i].shape) === 4) inputs[i].set(x);
    }

    const [gradX] = bwd.run(...inputs);
    for (let i = 0; i < 4; i++) {
      expect(gradX[i]).toBeCloseTo(1 / (x[i] * Math.LN2), 3);
    }
  });

  it('exp2 backward: grad * result * ln2', () => {
    const func = buildFunction('exp2_fwd', [t([4])], [t([4])], (b, args) => {
      b.returnOp([b.exp2(args[0]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);
    const x = [0, 1, 2, -1];
    const exp2X = x.map(v => Math.pow(2, v));
    const inputs = [];
    for (let i = 0; i < bwd.inputTypes.length; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 1, 1, 1]);

    let savedIdx = 1;
    for (let i = 1; i < inputs.length; i++) {
      if (numel(bwd.inputTypes[i].shape) === 4) {
        if (savedIdx === 1) {
          inputs[i].set(x);
          savedIdx++;
        } else {
          inputs[i].set(exp2X);
        }
      }
    }

    const [gradX] = bwd.run(...inputs);
    for (let i = 0; i < 4; i++) {
      expect(gradX[i]).toBeCloseTo(exp2X[i] * Math.LN2, 3);
    }
  });

  it('embedding backward: scatter-add gradients into weight rows', () => {
    const V = 5;
    const D = 3;
    const N = 4;
    const weightT = new TensorType([V, D], ScalarType.F32);
    const idxT = new TensorType([N], ScalarType.I32);
    const outT = new TensorType([N, D], ScalarType.F32);

    const func = buildFunction('emb_fwd', [weightT, idxT], [outT], (b, args) => {
      b.returnOp([b.embedding(args[0], args[1]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);

    const typedFor = (type, fill) => {
      const n = Math.max(numel(type.shape), 1);
      const arr = type.dtype === ScalarType.I32 ? new Int32Array(n) : new Float32Array(n);
      if (fill) fill(arr);
      return arr;
    };

    const indicesData = [2, 0, 2, 4];
    const gradOutData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) gradOutData[i] = i + 1;

    const inputs = bwd.inputTypes.map((type, i) => {
      if (type.shape.length === idxT.shape.length && type.dtype === ScalarType.I32) {
        return typedFor(type, a => a.set(indicesData));
      }
      if (type.shape.length === outT.shape.length &&
          type.shape[0] === outT.shape[0] && type.shape[1] === outT.shape[1]) {
        return typedFor(type, a => a.set(gradOutData));
      }
      return typedFor(type);
    });

    const outputs = bwd.run(...inputs);
    const gradWeight = outputs.find(buf => buf.length === V * D);
    if (!gradWeight) throw new Error('no V*D buffer in outputs');

    const expected = new Float32Array(V * D);
    for (let n = 0; n < N; n++) {
      const v = indicesData[n];
      for (let d = 0; d < D; d++) {
        expected[v * D + d] += gradOutData[n * D + d];
      }
    }
    for (let i = 0; i < V * D; i++) {
      expect(gradWeight[i]).toBeCloseTo(expected[i], 4);
    }
  });

  it('reshape backward: grad reshaped back', () => {
    const func = buildFunction('reshape_fwd', [t([2, 3])], [t([6])], (b, args) => {
      b.returnOp([b.reshape(args[0], [6]).getResult(0)]);
    });

    const bwd = buildAndCompileBackward(func);

    const inputs = [];
    for (let i = 0; i < bwd.inputTypes.length; i++) {
      inputs.push(new Float32Array(numel(bwd.inputTypes[i].shape)));
    }
    inputs[0].set([1, 2, 3, 4, 5, 6]);

    const [gradX] = bwd.run(...inputs);

    expect([...gradX]).toEqual([1, 2, 3, 4, 5, 6]);
    expect(bwd.outputTypes[0].shape).toEqual([2, 3]);
  });
});
