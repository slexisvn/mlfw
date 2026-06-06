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

describe('log(exp(x)) cancellation', () => {
  it('eliminates log(exp(x)) and returns x', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.log(b.exp(args[0]).getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('exp(log(x)) also cancels (both directions registered)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.log(args[0]).getResult(0)).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('log(x) alone is untouched', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.log(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('transpose(transpose(x)) composition', () => {
  it('composes two permutations into one', () => {
    const inT = new TensorType([2, 3, 4], ScalarType.F32);
    const midT = new TensorType([3, 2, 4], ScalarType.F32);
    const outT = new TensorType([4, 3, 2], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const t1 = b.transpose(args[0], [1, 0, 2]);
      b.returnOp([b.transpose(t1.getResult(0), [2, 0, 1]).getResult(0)]);
    });

    run(func);

    const liveTranspose = retVal(func).definingOp;
    expect(liveTranspose.opName).toBe('transpose');
    expect(liveTranspose.getOperand(0)).toBe(func.args[0]);

    const composed = liveTranspose.getAttr('permutation');
    expect(composed).toEqual([2, 1, 0]);
  });

  it('inverse transpose pair composes to identity permutation (fold is separate pass)', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const t1 = b.transpose(args[0], [2, 0, 1]);
      b.returnOp([b.transpose(t1.getResult(0), [1, 2, 0]).getResult(0)]);
    });

    run(func);

    const live = retVal(func).definingOp;
    expect(live.opName).toBe('transpose');
    expect(live.getOperand(0)).toBe(func.args[0]);
    expect(live.getAttr('permutation')).toEqual([0, 1, 2]);
  });

  it('three transposes collapse to one via iteration', () => {
    const t = new TensorType([2, 3, 4, 5], ScalarType.F32);
    const outT = new TensorType([5, 4, 3, 2], ScalarType.F32);
    const func = buildFunction('f', [t], [outT], (b, args) => {
      const t1 = b.transpose(args[0], [1, 0, 3, 2]);
      const t2 = b.transpose(t1.getResult(0), [3, 2, 1, 0]);
      b.returnOp([b.transpose(t2.getResult(0), [0, 1, 2, 3]).getResult(0)]);
    });

    run(func);

    const live = retVal(func).definingOp;
    expect(live.opName).toBe('transpose');
    expect(live.getOperand(0)).toBe(func.args[0]);
  });

  it('composition math is correct: newPerm[i] = perm1[perm2[i]]', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const func = buildFunction('f', [t], [new TensorType([3, 4, 2], ScalarType.F32)], (b, args) => {
      const t1 = b.transpose(args[0], [0, 2, 1]);
      b.returnOp([b.transpose(t1.getResult(0), [2, 1, 0]).getResult(0)]);
    });

    run(func);

    const perm = retVal(func).definingOp.getAttr('permutation');
    expect(perm).toEqual([1, 2, 0]);
    expect(retVal(func).type.shape).toEqual([3, 4, 2]);
  });
});

describe('convert(convert(x, dt1), dt2) → x when dt2 == original dtype', () => {
  it('eliminates roundtrip cast f32 → i32 → f32', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const toInt = b.convert(args[0], ScalarType.I32);
      b.returnOp([b.convert(toInt.getResult(0), ScalarType.F32).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('does NOT eliminate when final dtype differs from original', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F64);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const toInt = b.convert(args[0], ScalarType.I32);
      b.returnOp([b.convert(toInt.getResult(0), ScalarType.F64).getResult(0)]);
    });

    run(func);

    const outerConvert = retVal(func).definingOp;
    expect(outerConvert.opName).toBe('convert');
  });

  it('eliminates f64 → f32 → f64 roundtrip', () => {
    const t = new TensorType([4], ScalarType.F64);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const down = b.convert(args[0], ScalarType.F32);
      b.returnOp([b.convert(down.getResult(0), ScalarType.F64).getResult(0)]);
    });

    run(func);

    expect(retVal(func)).toBe(func.args[0]);
  });

  it('single convert is untouched', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.I32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      b.returnOp([b.convert(args[0], ScalarType.I32).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('pass behavior', () => {
  it('reaches fixed-point: second run returns UNCHANGED', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const na = b.neg(args[0]);
      const nb = b.neg(args[1]);
      const prod = b.mul(na.getResult(0), nb.getResult(0));
      b.returnOp([b.div(prod.getResult(0), prod.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('no dangling values after simplification', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const nb = b.neg(args[1]);
      const sum = b.add(args[0], nb.getResult(0));
      const toInt = b.convert(sum.getResult(0), ScalarType.I32);
      b.returnOp([b.convert(toInt.getResult(0), ScalarType.F32).getResult(0)]);
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
  });
});
