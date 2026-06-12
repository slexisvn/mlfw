import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { CanonicalizePass } from '../../../src/compiler/passes/canonicalize/canonicalize.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func) {
  return new CanonicalizePass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('exp(log(x)) in canonicalize — NOT cancelled (IEEE-unsound)', () => {
  it('exp(log(exp(log(x)))) stays (log of negative = NaN)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const inner = b.exp(b.log(args[0]).getResult(0));
      b.returnOp([b.exp(b.log(inner.getResult(0)).getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'exp')).not.toBeNull();
    expect(func.findOp(op => op.opName === 'log')).not.toBeNull();
  });
});

describe('reshape chain + trivial fold interaction', () => {
  it('reshape(reshape(x, mid), original) → identity via chain+trivial fold combo', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const r1 = b.reshape(args[0], [32]);
      b.returnOp([b.reshape(r1.getResult(0), [4, 8]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('4 chained reshapes collapse to one with final shape', () => {
    const inT = new TensorType([24], ScalarType.F32);
    const outT = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const r1 = b.reshape(args[0], [2, 12]);
      const r2 = b.reshape(r1.getResult(0), [3, 8]);
      const r3 = b.reshape(r2.getResult(0), [6, 4]);
      b.returnOp([b.reshape(r3.getResult(0), [4, 6]).getResult(0)]);
    });

    run(func);

    const live = retVal(func).definingOp;
    expect(live.opName).toBe('reshape');
    expect(live.getOperand(0)).toBe(func.args[0]);
    expect(live.getAttr('new_shape')).toEqual([4, 6]);
  });
});

describe('transpose + neg + add-zero multi-pattern', () => {
  it('identity transpose, double neg, and add-zero all fire to return x', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const transposed = b.transpose(args[0], [0, 1]);
      const nn = b.neg(b.neg(transposed.getResult(0)).getResult(0));
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.add(nn.getResult(0), zero.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });
});

describe('commutative swap enables further pattern', () => {
  it('add(0, x) → commutative swap → add(x, 0) → add-zero elimination → x', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.add(zero.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('mul(1, x) → commutative swap → mul(x, 1) → mul-one elimination → x', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.mul(one.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('mul(0, x) → commutative swap → mul(x, 0) → mul-zero annihilation (int only)', () => {
    const t = new TensorType([4], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.I32);
      b.returnOp([b.mul(zero.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'mul')).toBeNull();
    expect(retVal(func).type.dtype).toBe(ScalarType.I32);
  });
});

describe('sub-self on multi-dimensional', () => {
  it('sub(x,x) on 3D int tensor produces broadcast zero with correct shape', () => {
    const t = new TensorType([2, 3, 4], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[0]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'sub')).toBeNull();
    expect(retVal(func).type.shape).toEqual([2, 3, 4]);
  });
});

describe('div(x, 1) with higher-rank tensor', () => {
  it('div identity works on 3D tensors', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.div(args[0], one.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });
});
