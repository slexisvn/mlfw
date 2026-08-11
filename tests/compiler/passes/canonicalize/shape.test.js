import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CanonicalizePass } from '../../../../src/compiler/passes/canonicalize/canonicalize.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';

function run(func) {
  return new CanonicalizePass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('trivial reshape folding', () => {
  it('removes reshape when input and output shapes are identical', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.reshape(args[0], [4, 8]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
    expect(func.findOp(op => op.opName === 'reshape')).toBeNull();
  });

  it('keeps reshape when shapes actually differ', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [32]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('reshape chain composition', () => {
  it('collapses reshape(reshape(x)) — live reshape reads original input', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const r1 = b.reshape(args[0], [2, 16]);
      b.returnOp([b.reshape(r1.getResult(0), [32]).getResult(0)]);
    });

    run(func);

    const liveReshape = retVal(func).definingOp;
    expect(liveReshape.opName).toBe('reshape');
    expect(liveReshape.getAttr('new_shape')).toEqual([32]);
    expect(liveReshape.getOperand(0)).toBe(func.args[0]);
  });

  it('collapses 3 reshapes — live reshape reads original input with final shape', () => {
    const t1 = new TensorType([24], ScalarType.F32);
    const t4 = new TensorType([3, 8], ScalarType.F32);
    const func = buildFunction('f', [t1], [t4], (b, args) => {
      const r1 = b.reshape(args[0], [2, 12]);
      const r2 = b.reshape(r1.getResult(0), [4, 6]);
      b.returnOp([b.reshape(r2.getResult(0), [3, 8]).getResult(0)]);
    });

    run(func);

    const liveReshape = retVal(func).definingOp;
    expect(liveReshape.opName).toBe('reshape');
    expect(liveReshape.getOperand(0)).toBe(func.args[0]);
    expect(liveReshape.getAttr('new_shape')).toEqual([3, 8]);
  });

  it('reshape roundtrip folds to identity when final shape matches input', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const r1 = b.reshape(args[0], [32]);
      b.returnOp([b.reshape(r1.getResult(0), [4, 8]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });
});

describe('trivial transpose folding', () => {
  it('removes identity permutation transpose', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.transpose(args[0], [0, 1, 2]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('keeps non-identity permutation', () => {
    const inT = new TensorType([2, 3, 4], ScalarType.F32);
    const outT = new TensorType([4, 3, 2], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      b.returnOp([b.transpose(args[0], [2, 1, 0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('identity detection works for any rank', () => {
    for (const rank of [1, 2, 4, 5]) {
      const shape = Array.from({ length: rank }, (_, i) => i + 2);
      const t = new TensorType(shape, ScalarType.F32);
      const perm = Array.from({ length: rank }, (_, i) => i);
      const func = buildFunction('f', [t], [t], (b, args) => {
        b.returnOp([b.transpose(args[0], perm).getResult(0)]);
      });

      run(func);

      expect(func.findOp(op => op.opName === 'transpose')).toBeNull();
    }
  });
});

describe('cross-pattern interaction', () => {
  it('add-zero elimination followed by trivial reshape fold chains correctly', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const sum = b.add(args[0], zero.getResult(0));
      b.returnOp([b.reshape(sum.getResult(0), [4, 8]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });
});
