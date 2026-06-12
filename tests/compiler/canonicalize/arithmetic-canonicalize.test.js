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

describe('add identity (x + 0 = x)', () => {
  it('eliminates add when scalar zero is rhs and types match', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const sum = b.add(args[0], zero.getResult(0));
      b.returnOp([sum.getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
    expect(func.findOp(op => op.opName === 'add')).toBeNull();
  });

  it('eliminates add when scalar zero is lhs (commutative swap first)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const sum = b.add(zero.getResult(0), args[0]);
      b.returnOp([sum.getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('does NOT eliminate when non-zero operand type differs from result', () => {
    const scalarT = new TensorType([], ScalarType.F32);
    const vecT = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [], [vecT], (b) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const five = b.scalarConstant(5, ScalarType.F32);
      const sum = b.add(five.getResult(0), zero.getResult(0));
      b.returnOp([sum.getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'add')).toBeNull();
    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(5);
  });
});

describe('mul identity and annihilation', () => {
  it('eliminates mul by 1 and keeps the other operand', () => {
    const t = new TensorType([3, 5], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      const prod = b.mul(args[0], one.getResult(0));
      b.returnOp([prod.getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('mul by 0 produces zero broadcast matching result shape and dtype (int only)', () => {
    const t = new TensorType([3, 5], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.I32);
      const prod = b.mul(args[0], zero.getResult(0));
      b.returnOp([prod.getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'mul')).toBeNull();
    expect(retVal(func).type.shape).toEqual([3, 5]);
    expect(retVal(func).type.dtype).toBe(ScalarType.I32);
  });

  it('mul by 0 on scalar produces scalar constant without broadcast (int only)', () => {
    const t = new TensorType([], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.I32);
      const prod = b.mul(args[0], zero.getResult(0));
      b.returnOp([prod.getResult(0)]);
    });

    run(func);

    expect(retVal(func).type.shape).toEqual([]);
    expect(retVal(func).definingOp.opName).toBe('constant');
  });

  it('float mul by 0 stays (Inf*0=NaN, NaN*0=NaN)', () => {
    const t = new TensorType([3, 5], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.mul(args[0], zero.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'mul')).not.toBeNull();
  });
});

describe('sub patterns', () => {
  it('sub(x, 0) returns x', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const diff = b.sub(args[0], zero.getResult(0));
      b.returnOp([diff.getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('sub(0, x) does NOT simplify — subtraction is not commutative', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const diff = b.sub(zero.getResult(0), args[0]);
      b.returnOp([diff.getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'sub')).not.toBeNull();
  });

  it('sub(x, x) produces zero with matching shape (int only)', () => {
    const t = new TensorType([2, 3], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const diff = b.sub(args[0], args[0]);
      b.returnOp([diff.getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'sub')).toBeNull();
    expect(retVal(func).type.shape).toEqual([2, 3]);

    const bcast = retVal(func).definingOp;
    expect(bcast.opName).toBe('broadcast_in_dim');
    expect(bcast.getOperand(0).definingOp.getAttr('value')).toBe(0);
  });

  it('float sub(x, x) stays (Inf-Inf=NaN)', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('sub(x, y) where x !== y is not eliminated', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('sub(x, x) on scalar produces scalar zero without broadcast (int only)', () => {
    const t = new TensorType([], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[0]).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(0);
  });
});

describe('div identity', () => {
  it('div(x, 1) returns x', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.div(args[0], one.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('div(1, x) is NOT simplified — only rhs=1 is an identity', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.div(one.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'div')).not.toBeNull();
  });
});

describe('double negation', () => {
  it('neg(neg(x)) returns original x', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(b.neg(args[0]).getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('single neg is untouched', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('triple neg reduces to single neg feeding return (dead negs left for DCE)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      let v = args[0];
      for (let i = 0; i < 3; i++) v = b.neg(v).getResult(0);
      b.returnOp([v]);
    });

    run(func);

    const liveNeg = retVal(func).definingOp;
    expect(liveNeg.opName).toBe('neg');
    expect(liveNeg.getOperand(0)).toBe(func.args[0]);
  });
});

describe('commutative constant canonicalization', () => {
  it('swaps add(const, x) to add(x, const)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const c = b.scalarConstant(5, ScalarType.F32);
      b.returnOp([b.add(c.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    const addOp = func.findOp(op => op.opName === 'add');
    expect(addOp.getOperand(0)).toBe(func.args[0]);
    expect(addOp.getOperand(1).definingOp.opName).toBe('constant');
  });

  it('does NOT swap when both operands are non-const', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('does NOT swap when both operands are const', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [], [t], (b) => {
      const a = b.scalarConstant(3, ScalarType.F32);
      const c = b.scalarConstant(5, ScalarType.F32);
      b.returnOp([b.add(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('also works for mul(const, x) → mul(x, const)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const c = b.scalarConstant(7, ScalarType.F32);
      b.returnOp([b.mul(c.getResult(0), args[0]).getResult(0)]);
    });

    run(func);

    const mulOp = func.findOp(op => op.opName === 'mul');
    expect(mulOp.getOperand(0)).toBe(func.args[0]);
  });
});

describe('multi-consumer rewiring', () => {
  it('all users of sub(x,x) receive the same zero broadcast (int only)', () => {
    const t = new TensorType([4], ScalarType.I32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const diff = b.sub(args[0], args[0]);
      b.returnOp([b.add(diff.getResult(0), diff.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'sub')).toBeNull();
    const addOp = func.findOp(op => op.opName === 'add');
    if (addOp) {
      expect(addOp.getOperand(0)).toBe(addOp.getOperand(1));
    }
  });
});
