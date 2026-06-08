import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/backward_builder.js';
import { JointGraphBuilder } from '../../../src/compiler/ad/joint_builder.js';
import '../../../src/compiler/ad/index.js';

const F32 = ScalarType.F32;

function t(shape) {
  return new TensorType(shape, F32);
}

describe('BackwardGraphBuilder', () => {
  it('generates backward for add', () => {
    const func = buildFunction('add_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
      const result = b.add(args[0], args[1]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc, gradInputIndices } = builder.build(func);

    expect(backwardFunc.name).toBe('backward_add_fwd');
    expect(gradInputIndices).toEqual([0, 1]);
    expect(backwardFunc.outputTypes.length).toBe(2);

    const ops = backwardFunc.opsArray();
    const returnOp = ops.find(op => op.opName === 'return');
    expect(returnOp).toBeDefined();
    expect(returnOp.numOperands).toBe(2);
  });

  it('generates backward for mul', () => {
    const func = buildFunction('mul_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
      const result = b.mul(args[0], args[1]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const mulOps = ops.filter(op => op.opName === 'mul');
    expect(mulOps.length).toBeGreaterThanOrEqual(2);
  });

  it('generates backward for chained ops', () => {
    const func = buildFunction('chain_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
      const sum = b.add(args[0], args[1]).getResult(0);
      const result = b.mul(sum, args[0]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc, gradInputIndices } = builder.build(func);

    expect(gradInputIndices).toEqual([0, 1]);
    expect(backwardFunc.outputTypes.length).toBe(2);
  });

  it('handles gradient accumulation for shared inputs', () => {
    const func = buildFunction('shared_fwd', [t([4])], [t([4])], (b, args) => {
      const result = b.add(args[0], args[0]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const addOps = ops.filter(op => op.opName === 'add');
    expect(addOps.length).toBeGreaterThanOrEqual(1);
  });

  it('generates backward for exp', () => {
    const func = buildFunction('exp_fwd', [t([4])], [t([4])], (b, args) => {
      const result = b.exp(args[0]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const mulOps = ops.filter(op => op.opName === 'mul');
    expect(mulOps.length).toBeGreaterThanOrEqual(1);
  });

  it('generates backward for matmul', () => {
    const func = buildFunction('matmul_fwd', [t([2, 3]), t([3, 4])], [t([2, 4])], (b, args) => {
      const result = b.matmul(args[0], args[1]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const dotOps = ops.filter(op => op.opName === 'dot');
    expect(dotOps.length).toBeGreaterThanOrEqual(2);
  });

  it('generates backward for reduce sum', () => {
    const func = buildFunction('sum_fwd', [t([3, 4])], [t([3])], (b, args) => {
      const init = b.scalarConstant(0, F32).getResult(0);
      const result = b.reduce(args[0], init, [1], 'sum').getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const broadcastOps = ops.filter(op => op.opName === 'broadcast_in_dim');
    expect(broadcastOps.length).toBeGreaterThanOrEqual(1);
  });

  it('generates backward for reshape', () => {
    const func = buildFunction('reshape_fwd', [t([2, 3])], [t([6])], (b, args) => {
      const result = b.reshape(args[0], [6]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const reshapeOps = ops.filter(op => op.opName === 'reshape');
    expect(reshapeOps.length).toBeGreaterThanOrEqual(1);
  });

  it('generates backward for sigmoid', () => {
    const func = buildFunction('sig_fwd', [t([4])], [t([4])], (b, args) => {
      const result = b.sigmoid(args[0]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new BackwardGraphBuilder();
    const { backwardFunc } = builder.build(func);

    const ops = backwardFunc.opsArray();
    const mulOps = ops.filter(op => op.opName === 'mul');
    expect(mulOps.length).toBeGreaterThanOrEqual(2);
  });
});

describe('JointGraphBuilder', () => {
  it('creates joint graph for add', () => {
    const func = buildFunction('add_fwd', [t([4]), t([4])], [t([4])], (b, args) => {
      const result = b.add(args[0], args[1]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new JointGraphBuilder();
    const { jointFunc, numForwardOutputs, numGradInputs } = builder.build(func);

    expect(jointFunc.name).toBe('joint_add_fwd');
    expect(numForwardOutputs).toBe(1);
    expect(numGradInputs).toBe(2);
    expect(jointFunc.inputTypes.length).toBe(3);
    expect(jointFunc.outputTypes.length).toBe(3);
  });

  it('creates joint graph for matmul', () => {
    const func = buildFunction('matmul_fwd', [t([2, 3]), t([3, 4])], [t([2, 4])], (b, args) => {
      const result = b.matmul(args[0], args[1]).getResult(0);
      b.returnOp([result]);
    });

    const builder = new JointGraphBuilder();
    const { jointFunc, numForwardOutputs, numGradInputs } = builder.build(func);

    expect(numForwardOutputs).toBe(1);
    expect(numGradInputs).toBe(2);
    expect(jointFunc.inputTypes.length).toBe(3);
  });
});
