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

describe('exp(log(x)) cancellation', () => {
  it('eliminates exp(log(x)) and returns original x', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.log(args[0]).getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('lone exp is untouched', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('log(exp(x)) is NOT cancelled — only exp∘log pattern is registered', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.log(b.exp(args[0]).getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'log')).not.toBeNull();
    expect(func.findOp(op => op.opName === 'exp')).not.toBeNull();
  });
});

describe('fixed-point convergence', () => {
  it('second pass returns UNCHANGED — graph already canonical', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.add(b.neg(b.neg(args[0]).getResult(0)).getResult(0),
        zero.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('neg(neg) + add(_, 0) chain collapses to just x', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const nn = b.neg(b.neg(args[0]).getResult(0));
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.add(nn.getResult(0), zero.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('quad neg collapses to identity', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      let v = args[0];
      for (let i = 0; i < 4; i++) v = b.neg(v).getResult(0);
      b.returnOp([v]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });
});

describe('fold transpose into dot', () => {
  it('folds transpose on rhs — dot reads original operand with swapped contracting', () => {
    const lhs = new TensorType([3, 5], ScalarType.F32);
    const rhs = new TensorType([5, 7], ScalarType.F32);
    const out = new TensorType([3, 7], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      const t = b.transpose(args[1], [1, 0]);
      b.returnOp([b.dot(args[0], t.getResult(0), [1], [0]).getResult(0)]);
    });

    run(func);

    const dot = retVal(func).definingOp;
    expect(dot.opName).toBe('dot');
    expect(dot.getOperand(1)).toBe(func.args[1]);
    expect(dot.getAttr('rhs_contracting')).toEqual([1]);
  });

  it('folds transpose on lhs — dot reads original operand with swapped contracting', () => {
    const lhs = new TensorType([5, 3], ScalarType.F32);
    const rhs = new TensorType([5, 7], ScalarType.F32);
    const out = new TensorType([3, 7], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      const t = b.transpose(args[0], [1, 0]);
      b.returnOp([b.dot(t.getResult(0), args[1], [1], [0]).getResult(0)]);
    });

    run(func);

    const dot = retVal(func).definingOp;
    expect(dot.opName).toBe('dot');
    expect(dot.getOperand(0)).toBe(func.args[0]);
    expect(dot.getAttr('lhs_contracting')).toEqual([0]);
  });

  it('dot without transpose operands is untouched', () => {
    const lhs = new TensorType([3, 5], ScalarType.F32);
    const rhs = new TensorType([5, 7], ScalarType.F32);
    const out = new TensorType([3, 7], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.dot(args[0], args[1], [1], [0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('pass behavior', () => {
  it('returns UNCHANGED on graph with no canonicalization opportunities', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(b.add(args[0], args[1]).getResult(0), args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('exp survives when neg(neg) around it is removed', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(b.neg(b.exp(args[0]).getResult(0)).getResult(0)).getResult(0)]);
    });

    run(func);

    const expOp = func.findOp(op => op.opName === 'exp');
    expect(expOp).not.toBeNull();
    expect(expOp.getOperand(0)).toBe(func.args[0]);
    expect(retVal(func)).toBe(expOp.getResult(0));
  });

  it('no dangling values after multi-pattern canonicalization', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const nn = b.neg(b.neg(args[0]).getResult(0));
      const zero = b.scalarConstant(0, ScalarType.F32);
      const sum = b.add(nn.getResult(0), zero.getResult(0));
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.mul(sum.getResult(0), one.getResult(0)).getResult(0)]);
    });

    run(func);

    for (const op of func.ops()) {
      for (let i = 0; i < op.numOperands; i++) {
        const v = op.getOperand(i);
        if (!v.isBlockArgument()) {
          expect(v.definingOp).not.toBeNull();
          expect(v.definingOp.parentBlock).not.toBeNull();
        }
      }
    }

    expect(retVal(func)).toBe(func.args[0]);
  });
});
