import { describe, it, expect, afterEach } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { DecompositionPass, registerDecomposition, unregisterDecomposition } from '../../../../src/compiler/passes/decompose/decomposition_pass.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';

const CUSTOM_OP = '__test_custom_decompose__';

afterEach(() => unregisterDecomposition(CUSTOM_OP));

function run(func) {
  return new DecompositionPass().run(func);
}

describe('where decomposition', () => {
  it('skips compare when condition is already bool', () => {
    const boolT = new TensorType([4, 4], ScalarType.BOOL);
    const floatT = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('f', [boolT, floatT, floatT], [floatT], (b, args) => {
      b.returnOp([b.where(args[0], args[1], args[2]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'compare')).toBeNull();
    const sel = func.findOp(op => op.opName === 'select');
    expect(sel.getOperand(0)).toBe(func.args[0]);
  });

  it('inserts ne-zero compare when condition is integer', () => {
    const intT = new TensorType([4, 4], ScalarType.I32);
    const floatT = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('f', [intT, floatT, floatT], [floatT], (b, args) => {
      b.returnOp([b.where(args[0], args[1], args[2]).getResult(0)]);
    });

    run(func);

    const cmp = func.findOp(op => op.opName === 'compare');
    expect(cmp).not.toBeNull();
    expect(cmp.getAttr('direction')).toBe('ne');
    expect(cmp.getResult(0).type.dtype).toBe(ScalarType.BOOL);

    const sel = func.findOp(op => op.opName === 'select');
    expect(sel.getOperand(0)).toBe(cmp.getResult(0));
  });

  it('preserves x and y operand identity through select', () => {
    const boolT = new TensorType([2, 3], ScalarType.BOOL);
    const floatT = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('f', [boolT, floatT, floatT], [floatT], (b, args) => {
      b.returnOp([b.where(args[0], args[1], args[2]).getResult(0)]);
    });

    run(func);

    const sel = func.findOp(op => op.opName === 'select');
    expect(sel.getOperand(1)).toBe(func.args[1]);
    expect(sel.getOperand(2)).toBe(func.args[2]);
  });
});

describe('split decomposition', () => {
  it('computes slice offsets cumulatively along split dimension', () => {
    const t = new TensorType([10, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [
      new TensorType([3, 4], ScalarType.F32),
      new TensorType([3, 4], ScalarType.F32),
      new TensorType([4, 4], ScalarType.F32),
    ], (b, args) => {
      const s = b.split(args[0], 0, [3, 3, 4]);
      b.returnOp([s.getResult(0), s.getResult(1), s.getResult(2)]);
    });

    run(func);

    const slices = func.findOps(op => op.opName === 'slice')
      .sort((a, b) => a.getAttr('starts')[0] - b.getAttr('starts')[0]);

    expect(slices[0].getAttr('starts')).toEqual([0, 0]);
    expect(slices[0].getAttr('limits')).toEqual([3, 4]);

    expect(slices[1].getAttr('starts')).toEqual([3, 0]);
    expect(slices[1].getAttr('limits')).toEqual([6, 4]);

    expect(slices[2].getAttr('starts')).toEqual([6, 0]);
    expect(slices[2].getAttr('limits')).toEqual([10, 4]);
  });

  it('offsets shift on correct dimension when splitting dim=1', () => {
    const t = new TensorType([4, 12], ScalarType.F32);
    const func = buildFunction('f', [t], [
      new TensorType([4, 5], ScalarType.F32),
      new TensorType([4, 7], ScalarType.F32),
    ], (b, args) => {
      const s = b.split(args[0], 1, [5, 7]);
      b.returnOp([s.getResult(0), s.getResult(1)]);
    });

    run(func);

    const slices = func.findOps(op => op.opName === 'slice')
      .sort((a, b) => a.getAttr('starts')[1] - b.getAttr('starts')[1]);

    expect(slices[0].getAttr('starts')[0]).toBe(0);
    expect(slices[0].getAttr('starts')[1]).toBe(0);
    expect(slices[0].getAttr('limits')[0]).toBe(4);
    expect(slices[0].getAttr('limits')[1]).toBe(5);

    expect(slices[1].getAttr('starts')[0]).toBe(0);
    expect(slices[1].getAttr('starts')[1]).toBe(5);
    expect(slices[1].getAttr('limits')[1]).toBe(12);
  });

  it('each slice reads from the original input, not from a previous slice', () => {
    const t = new TensorType([9, 3], ScalarType.F32);
    const func = buildFunction('f', [t], [
      new TensorType([3, 3], ScalarType.F32),
      new TensorType([3, 3], ScalarType.F32),
      new TensorType([3, 3], ScalarType.F32),
    ], (b, args) => {
      const s = b.split(args[0], 0, [3, 3, 3]);
      b.returnOp([s.getResult(0), s.getResult(1), s.getResult(2)]);
    });

    run(func);

    const slices = func.findOps(op => op.opName === 'slice');
    for (const s of slices) {
      expect(s.getOperand(0)).toBe(func.args[0]);
    }
  });
});

describe('one_hot decomposition', () => {
  it('resolves negative axis before creating iota', () => {
    const idxT = new TensorType([4], ScalarType.I32);
    const outT = new TensorType([4, 10], ScalarType.F32);
    const func = buildFunction('f', [idxT], [outT], (b, args) => {
      b.returnOp([b.oneHot(args[0], 10, { dtype: ScalarType.F32 }).getResult(0)]);
    });

    run(func);

    const iota = func.findOp(op => op.opName === 'iota');
    expect(iota.getAttr('iota_dimension')).toBe(1);
  });

  it('on/off values match when custom values are specified', () => {
    const idxT = new TensorType([3], ScalarType.I32);
    const outT = new TensorType([3, 5], ScalarType.F32);
    const func = buildFunction('f', [idxT], [outT], (b, args) => {
      b.returnOp([b.oneHot(args[0], 5, { onValue: 7, offValue: -1, dtype: ScalarType.F32 }).getResult(0)]);
    });

    run(func);

    const constants = func.findOps(op => op.opName === 'constant');
    const values = constants.map(c => c.getAttr('value'));
    expect(values).toContain(7);
    expect(values).toContain(-1);
  });
});

describe('embedding decomposition', () => {
  it('gather slice_sizes matches [1, embedding_dim] from weight shape', () => {
    const weightT = new TensorType([1000, 64], ScalarType.F32);
    const idxT = new TensorType([4, 8], ScalarType.I32);
    const outT = new TensorType([4, 8, 64], ScalarType.F32);
    const func = buildFunction('f', [weightT, idxT], [outT], (b, args) => {
      b.returnOp([b.embedding(args[0], args[1]).getResult(0)]);
    });

    run(func);

    const gather = func.findOp(op => op.opName === 'gather');
    expect(gather.getAttr('slice_sizes')).toEqual([1, 64]);
    expect(gather.getOperand(0)).toBe(func.args[0]);
    expect(gather.getOperand(1)).toBe(func.args[1]);
  });

  it('index_vector_dim equals indices rank', () => {
    const weightT = new TensorType([500, 128], ScalarType.F32);
    const idx1D = new TensorType([16], ScalarType.I32);
    const func1 = buildFunction('f1', [weightT, idx1D],
      [new TensorType([16, 128], ScalarType.F32)], (b, args) => {
        b.returnOp([b.embedding(args[0], args[1]).getResult(0)]);
      });

    const idx3D = new TensorType([2, 4, 8], ScalarType.I32);
    const func3 = buildFunction('f3', [weightT, idx3D],
      [new TensorType([2, 4, 8, 128], ScalarType.F32)], (b, args) => {
        b.returnOp([b.embedding(args[0], args[1]).getResult(0)]);
      });

    run(func1);
    run(func3);

    expect(func1.findOp(op => op.opName === 'gather').getAttr('index_vector_dim')).toBe(1);
    expect(func3.findOp(op => op.opName === 'gather').getAttr('index_vector_dim')).toBe(3);
  });
});

describe('pass edge cases', () => {
  it('returns UNCHANGED and does not modify graph when no composites exist', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const opsBefore = func.numOps();
    const result = run(func);

    expect(result).toBe(PassResult.UNCHANGED);
    expect(func.numOps()).toBe(opsBefore);
  });

  it('custom registered decomposition gets picked up by the pass', () => {
    const customName = CUSTOM_OP;

    registerDecomposition(customName, (op, b) => {
      const x = op.getOperand(0);
      const doubled = b.add(x, x);
      op.replaceAllResultsWith([doubled.getResult(0)]);
      op.erase();
    });

    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const op = b._inferAndBuild(customName, [args[0]], null, null, [t]);
      b.returnOp([op.getResult(0)]);
    });

    const result = run(func);

    expect(result).toBe(PassResult.CHANGED);
    expect(func.findOp(op => op.opName === customName)).toBeNull();

    const addOp = func.findOp(op => op.opName === 'add');
    expect(addOp.getOperand(0)).toBe(func.args[0]);
    expect(addOp.getOperand(1)).toBe(func.args[0]);
  });
});
