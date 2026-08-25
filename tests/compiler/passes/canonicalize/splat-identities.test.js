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

function splat(value, count, Ctor = Float32Array) {
  const data = new Ctor(count);
  data.fill(value);
  return data;
}

describe('mul(x, 1) folds whether the 1 is a scalar or a shaped splat', () => {
  it('folds a scalar constant 1', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.mul(args[0], one.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
    expect(func.findOp(op => op.opName === 'mul')).toBeNull();
  });

  it('folds a dense constant whose every element is 1', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const ones = b.tensorConstant(splat(1, 4), [4], ScalarType.F32);
      b.returnOp([b.mul(args[0], ones.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
    expect(func.findOp(op => op.opName === 'mul')).toBeNull();
  });

  it('folds a shaped constant carrying a single splat value', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const ones = b.constant(1, new TensorType([4], ScalarType.F32));
      b.returnOp([b.mul(args[0], ones.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('folds through a broadcast of a scalar 1', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      const spread = b.broadcast(one.getResult(0), [4], []);
      b.returnOp([b.mul(args[0], spread.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
    expect(func.findOp(op => op.opName === 'mul')).toBeNull();
  });

  it('folds an integer splat held as a BigInt64Array', () => {
    const t = new TensorType([3], ScalarType.I64);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const ones = b.tensorConstant(splat(1n, 3, BigInt64Array), [3], ScalarType.I64);
      b.returnOp([b.mul(args[0], ones.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('does NOT fold a dense constant that is not uniform', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const mixed = b.tensorConstant(new Float32Array([1, 1, 2, 1]), [4], ScalarType.F32);
      b.returnOp([b.mul(args[0], mixed.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('mul');
  });

  it('does NOT fold a dense constant of some other uniform value', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const threes = b.tensorConstant(splat(3, 4), [4], ScalarType.F32);
      b.returnOp([b.mul(args[0], threes.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('mul');
  });

  it('does NOT fold an empty dense constant, which asserts nothing about its elements', () => {
    const t = new TensorType([0], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const empty = b.tensorConstant(new Float32Array(0), [0], ScalarType.F32);
      b.returnOp([b.mul(args[0], empty.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('mul');
  });
});

describe('div(x, 1) and sub(x, 0) accept the splat form too', () => {
  it('div by a dense one folds to x', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const ones = b.tensorConstant(splat(1, 6), [2, 3], ScalarType.F32);
      b.returnOp([b.div(args[0], ones.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('div by a dense value other than one stays', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const twos = b.tensorConstant(splat(2, 6), [2, 3], ScalarType.F32);
      b.returnOp([b.div(args[0], twos.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('div');
  });

  it('sub of a dense zero folds to x, which is exact for floats including -0', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zeros = b.tensorConstant(splat(0, 4), [4], ScalarType.F32);
      b.returnOp([b.sub(args[0], zeros.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });
});

describe('a splat operand does not weaken the dtype and shape rules', () => {
  it('add of a dense zero folds on integers', () => {
    const t = new TensorType([4], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zeros = b.tensorConstant(splat(0, 4, Int32Array), [4], ScalarType.I32);
      b.returnOp([b.add(args[0], zeros.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('add of a dense zero stays on floats, where x + 0 loses the sign of -0', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zeros = b.tensorConstant(splat(0, 4), [4], ScalarType.F32);
      b.returnOp([b.add(args[0], zeros.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('add');
  });

  it('mul by a dense zero stays on floats, where Inf * 0 is NaN', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zeros = b.tensorConstant(splat(0, 4), [4], ScalarType.F32);
      b.returnOp([b.mul(args[0], zeros.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('mul');
  });

  it('does NOT fold when the kept operand is narrower than the broadcast result', () => {
    const scalarT = new TensorType([], ScalarType.F32);
    const vecT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [scalarT], [vecT], (b, args) => {
      const ones = b.tensorConstant(splat(1, 4), [4], ScalarType.F32);
      b.returnOp([b.mul(args[0], ones.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('mul');
    expect(retVal(func).type.equals(vecT)).toBe(true);
  });

  it('does NOT fold a sub whose dense zero is on the left', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zeros = b.tensorConstant(splat(0, 4), [4], ScalarType.F32);
      b.returnOp([b.sub(zeros.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('sub');
  });
});
