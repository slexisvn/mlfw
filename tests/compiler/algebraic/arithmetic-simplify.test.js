import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { AlgebraicSimplificationPass } from '../../../src/compiler/passes/simplify/algebraic.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func) {
  return new AlgebraicSimplificationPass().run(func);
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('div(x, x) → 1', () => {
  it('replaces with broadcast(1) matching result shape', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.div(args[0], args[0]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'div')).toBeNull();
    expect(retVal(func).type.shape).toEqual([4, 8]);
    const bcast = retVal(func).definingOp;
    expect(bcast.opName).toBe('broadcast_in_dim');
    expect(bcast.getOperand(0).definingOp.getAttr('value')).toBe(1);
  });

  it('produces scalar constant 1 without broadcast on scalar input', () => {
    const t = new TensorType([], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.div(args[0], args[0]).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('constant');
    expect(retVal(func).definingOp.getAttr('value')).toBe(1);
  });

  it('uses result dtype, not hardcoded f32', () => {
    const t = new TensorType([4], ScalarType.F64);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.div(args[0], args[0]).getResult(0)]);
    });

    run(func);

    expect(retVal(func).type.dtype).toBe(ScalarType.F64);
    const constOp = func.findOp(op => op.opName === 'constant');
    expect(constOp.getResult(0).type.dtype).toBe(ScalarType.F64);
  });

  it('div(x, y) where x !== y is NOT simplified', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.div(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('mul(neg(a), neg(b)) → mul(a, b)', () => {
  it('strips both negations and muls the original values', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const na = b.neg(args[0]);
      const nb = b.neg(args[1]);
      b.returnOp([b.mul(na.getResult(0), nb.getResult(0)).getResult(0)]);
    });

    run(func);

    const liveMul = retVal(func).definingOp;
    expect(liveMul.opName).toBe('mul');
    expect(liveMul.getOperand(0)).toBe(func.args[0]);
    expect(liveMul.getOperand(1)).toBe(func.args[1]);
  });

  it('does NOT fire when only one operand is neg', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const na = b.neg(args[0]);
      b.returnOp([b.mul(na.getResult(0), args[1]).getResult(0)]);
    });

    run(func);

    const mul = retVal(func).definingOp;
    expect(mul.opName).toBe('mul');
    expect(mul.getOperand(0).definingOp.opName).toBe('neg');
  });

  it('works with self-neg: mul(neg(x), neg(x)) → mul(x, x)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n = b.neg(args[0]);
      b.returnOp([b.mul(n.getResult(0), n.getResult(0)).getResult(0)]);
    });

    run(func);

    const mul = retVal(func).definingOp;
    expect(mul.opName).toBe('mul');
    expect(mul.getOperand(0)).toBe(func.args[0]);
    expect(mul.getOperand(1)).toBe(func.args[0]);
  });
});

describe('add(a, neg(b)) → sub(a, b)', () => {
  it('converts addition of negation into subtraction', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const nb = b.neg(args[1]);
      b.returnOp([b.add(args[0], nb.getResult(0)).getResult(0)]);
    });

    run(func);

    const liveSub = retVal(func).definingOp;
    expect(liveSub.opName).toBe('sub');
    expect(liveSub.getOperand(0)).toBe(func.args[0]);
    expect(liveSub.getOperand(1)).toBe(func.args[1]);
  });

  it('only matches neg on rhs — add(neg(a), b) does NOT fire', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const na = b.neg(args[0]);
      b.returnOp([b.add(na.getResult(0), args[1]).getResult(0)]);
    });

    run(func);

    expect(retVal(func).definingOp.opName).toBe('add');
  });
});

describe('sub(a, neg(b)) → add(a, b)', () => {
  it('converts subtraction of negation into addition', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const nb = b.neg(args[1]);
      b.returnOp([b.sub(args[0], nb.getResult(0)).getResult(0)]);
    });

    run(func);

    const liveAdd = retVal(func).definingOp;
    expect(liveAdd.opName).toBe('add');
    expect(liveAdd.getOperand(0)).toBe(func.args[0]);
    expect(liveAdd.getOperand(1)).toBe(func.args[1]);
  });

  it('does NOT fire when rhs is not neg', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('cross-pattern chains', () => {
  it('sub(a, neg(b)) → add then add(a, neg(c)) on the result chains correctly', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, t], [t], (b, args) => {
      const nb = b.neg(args[1]);
      const step1 = b.sub(args[0], nb.getResult(0));
      const nc = b.neg(args[2]);
      b.returnOp([b.add(step1.getResult(0), nc.getResult(0)).getResult(0)]);
    });

    run(func);

    const outerSub = retVal(func).definingOp;
    expect(outerSub.opName).toBe('sub');
    expect(outerSub.getOperand(1)).toBe(func.args[2]);

    const innerAdd = outerSub.getOperand(0).definingOp;
    expect(innerAdd.opName).toBe('add');
    expect(innerAdd.getOperand(0)).toBe(func.args[0]);
    expect(innerAdd.getOperand(1)).toBe(func.args[1]);
  });

  it('mul(neg(a), neg(b)) then div result by itself → constant 1', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const na = b.neg(args[0]);
      const nb = b.neg(args[1]);
      const prod = b.mul(na.getResult(0), nb.getResult(0));
      b.returnOp([b.div(prod.getResult(0), prod.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'div')).toBeNull();
    const constOp = func.findOps(op => op.opName === 'constant')
      .find(c => c.getAttr('value') === 1);
    expect(constOp).toBeDefined();
  });
});
