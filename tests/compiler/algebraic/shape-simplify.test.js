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

describe('ReshapeReshape in algebraic pass', () => {
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
    expect(liveReshape.getOperand(0)).toBe(func.args[0]);
    expect(liveReshape.getAttr('new_shape')).toEqual([32]);
  });

  it('three reshapes collapse via iteration to one reading original input', () => {
    const t1 = new TensorType([24], ScalarType.F32);
    const tOut = new TensorType([2, 12], ScalarType.F32);
    const func = buildFunction('f', [t1], [tOut], (b, args) => {
      const r1 = b.reshape(args[0], [6, 4]);
      const r2 = b.reshape(r1.getResult(0), [3, 8]);
      b.returnOp([b.reshape(r2.getResult(0), [2, 12]).getResult(0)]);
    });

    run(func);

    const live = retVal(func).definingOp;
    expect(live.opName).toBe('reshape');
    expect(live.getOperand(0)).toBe(func.args[0]);
    expect(live.getAttr('new_shape')).toEqual([2, 12]);
  });
});

describe('cross-pattern: reshape + arithmetic in algebraic', () => {
  it('div(x,x) and reshape(reshape) both fire in same pass', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [new TensorType([32], ScalarType.F32)], (b, args) => {
      const d = b.div(args[0], args[0]);
      const r1 = b.reshape(d.getResult(0), [2, 16]);
      b.returnOp([b.reshape(r1.getResult(0), [32]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'div')).toBeNull();

    const live = retVal(func).definingOp;
    expect(live.opName).toBe('reshape');
    expect(live.getAttr('new_shape')).toEqual([32]);
  });
});

describe('AddZero and friends in algebraic context', () => {
  it('add(x, 0) eliminates in algebraic pass (not just canonicalize)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.add(args[0], zero.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('sub(x, x) produces zero in algebraic pass', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[0]).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'sub')).toBeNull();
  });

  it('mul(x, 0) annihilates in algebraic pass', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.mul(args[0], zero.getResult(0)).getResult(0)]);
    });

    run(func);

    expect(func.findOp(op => op.opName === 'mul')).toBeNull();
  });
});

describe('complex cross-pattern chain in algebraic', () => {
  it('add(neg(a), neg(b)) → sub then SubNegToAdd if b is neg of something', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t, t], [t], (b, args) => {
      const nb = b.neg(args[1]);
      const nc = b.neg(args[2]);
      const sum = b.add(args[0], nb.getResult(0));
      b.returnOp([b.sub(sum.getResult(0), nc.getResult(0)).getResult(0)]);
    });

    run(func);

    const ret = retVal(func).definingOp;
    expect(ret.opName).toBe('add');
    expect(ret.getOperand(1)).toBe(func.args[2]);

    const inner = ret.getOperand(0).definingOp;
    expect(inner.opName).toBe('sub');
    expect(inner.getOperand(0)).toBe(func.args[0]);
    expect(inner.getOperand(1)).toBe(func.args[1]);
  });
});
