import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { AlgebraicSimplificationPass } from '../../../../src/compiler/passes/simplify/algebraic.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { roundToDtype } from '../../../../src/tensor/utils/half.js';

function run(func) {
  return new AlgebraicSimplificationPass().run(func);
}

function runFast(func) {
  return new AlgebraicSimplificationPass({ fastMath: true }).run(func);
}

function roundTrip(origDtype, midDtype, shape = [4], name = 'f') {
  const t = new TensorType(shape, origDtype);
  return buildFunction(name, [t], [t], (b, args) => {
    const mid = b.convert(args[0], midDtype);
    b.returnOp([b.convert(mid.getResult(0), origDtype).getResult(0)]);
  });
}

function retVal(func) {
  return func.getReturnOp().getOperand(0);
}

describe('log(exp(x)) / exp(log(x)) are NOT cancelled (IEEE-unsound)', () => {
  it('log(exp(x)) stays (exp overflow → log(Inf)=Inf, not x)', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.log(b.exp(args[0]).getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('log');
  });

  it('exp(log(x)) stays (x<0 → log(x)=NaN, not x)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.log(args[0]).getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('exp');
  });

  it('log(x) alone is untouched', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.log(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('fastMath gate re-enables IEEE-unsound rewrites only when opted in', () => {
  function runFast(func) {
    return new AlgebraicSimplificationPass({ fastMath: true }).run(func);
  }

  it('exp(log(x)) → x under fastMath', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(b.log(args[0]).getResult(0)).getResult(0)]);
    });
    expect(runFast(func)).toBe(PassResult.CHANGED);
    expect(retVal(func)).toBe(func.entryBlock.arguments[0]);
  });

  it('log(exp(x)) → x under fastMath', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.log(b.exp(args[0]).getResult(0)).getResult(0)]);
    });
    expect(runFast(func)).toBe(PassResult.CHANGED);
    expect(retVal(func)).toBe(func.entryBlock.arguments[0]);
  });

  it('div(x, x) → 1 under fastMath, untouched by default', () => {
    const t = new TensorType([4], ScalarType.F32);
    const make = () => buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.div(args[0], args[0]).getResult(0)]);
    });
    const def = make();
    expect(run(def)).toBe(PassResult.UNCHANGED);
    expect(retVal(def).definingOp.opName).toBe('div');

    const fast = make();
    expect(runFast(fast)).toBe(PassResult.CHANGED);
    expect(retVal(fast).definingOp.opName).not.toBe('div');
  });

  it('float add(x, 0), sub(x, x) and mul(x, 0) stay by default, fold under fastMath', () => {
    const t = new TensorType([4], ScalarType.F32);
    const zero = (b) => b.scalarConstant(0, ScalarType.F32).getResult(0);
    const bodies = {
      'add(x, 0)': (b, args) => b.returnOp([b.add(args[0], zero(b)).getResult(0)]),
      'sub(x, x)': (b, args) => b.returnOp([b.sub(args[0], args[0]).getResult(0)]),
      'mul(x, 0)': (b, args) => b.returnOp([b.mul(args[0], zero(b)).getResult(0)]),
    };

    for (const [name, body] of Object.entries(bodies)) {
      expect(run(buildFunction('f', [t], [t], body)), name).toBe(PassResult.UNCHANGED);
      expect(runFast(buildFunction('f', [t], [t], body)), name).toBe(PassResult.CHANGED);
    }
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

describe('convert round trips fold only when the middle dtype holds every source value', () => {
  const exact = [
    ['f16 → f32 → f16', ScalarType.F16, ScalarType.F32],
    ['bf16 → f32 → bf16', ScalarType.BF16, ScalarType.F32],
    ['f32 → f64 → f32', ScalarType.F32, ScalarType.F64],
    ['i8 → i32 → i8', ScalarType.I8, ScalarType.I32],
    ['i16 → f32 → i16', ScalarType.I16, ScalarType.F32],
  ];

  for (const [label, orig, mid] of exact) {
    it(`folds ${label}: the middle dtype represents every source value, so nothing is lost`, () => {
      const func = roundTrip(orig, mid);

      expect(run(func)).toBe(PassResult.CHANGED);
      expect(retVal(func)).toBe(func.args[0]);
    });
  }

  const lossyFloat = [
    ['f32 → f16 → f32', ScalarType.F32, ScalarType.F16],
    ['f32 → bf16 → f32', ScalarType.F32, ScalarType.BF16],
    ['f64 → f32 → f64', ScalarType.F64, ScalarType.F32],
    ['f16 → bf16 → f16', ScalarType.F16, ScalarType.BF16],
  ];

  for (const [label, orig, mid] of lossyFloat) {
    it(`keeps ${label} by default and drops it under fastMath: the middle dtype rounds`, () => {
      const kept = roundTrip(orig, mid);
      expect(run(kept)).toBe(PassResult.UNCHANGED);
      expect(retVal(kept).definingOp.opName).toBe('convert');

      const fast = roundTrip(orig, mid);
      expect(runFast(fast)).toBe(PassResult.CHANGED);
      expect(retVal(fast)).toBe(fast.args[0]);
    });
  }

  const truncating = [
    ['f32 → i32 → f32', ScalarType.F32, ScalarType.I32],
    ['i32 → i8 → i32', ScalarType.I32, ScalarType.I8],
    ['i32 → f32 → i32', ScalarType.I32, ScalarType.F32],
    ['i64 → f64 → i64', ScalarType.I64, ScalarType.F64],
  ];

  for (const [label, orig, mid] of truncating) {
    it(`keeps ${label} even under fastMath: the middle dtype discards whole values, not ulps`, () => {
      const kept = roundTrip(orig, mid);
      expect(run(kept)).toBe(PassResult.UNCHANGED);

      const fast = roundTrip(orig, mid);
      expect(runFast(fast)).toBe(PassResult.UNCHANGED);
      expect(retVal(fast).definingOp.opName).toBe('convert');
    });
  }

  it('leaves a pair whose final dtype differs from the original alone', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F64);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const mid = b.convert(args[0], ScalarType.F16);
      b.returnOp([b.convert(mid.getResult(0), ScalarType.F64).getResult(0)]);
    });

    expect(runFast(func)).toBe(PassResult.UNCHANGED);
    expect(retVal(func).definingOp.opName).toBe('convert');
  });

  it('leaves a lone convert alone', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F16);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      b.returnOp([b.convert(args[0], ScalarType.F16).getResult(0)]);
    });

    expect(runFast(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('a compiled f32 → f16 → f32 round trip keeps the f16 rounding', () => {
  const X = new Float32Array([1.1, 0.1, 2.5, 65536, 1e-8]);

  function compiled(fastMath) {
    const func = roundTrip(ScalarType.F32, ScalarType.F16, [X.length], 'half_trip');
    const out = new Float32Array(X.length);
    compileGraph(func, CPUTarget(), { optimization: { fastMath } }).run('half_trip', X, out);
    return out;
  }

  it('returns each input rounded to f16, including the overflow to Infinity', () => {
    const out = compiled(false);

    for (let i = 0; i < X.length; i++) expect(out[i]).toBe(roundToDtype('f16', X[i]));
    expect(out[0]).not.toBe(X[0]);
    expect(out[2]).toBe(X[2]);
    expect(out[3]).toBe(Infinity);
    expect(out[4]).toBe(0);
  });

  it('returns the inputs untouched under fastMath, where the round trip is folded away', () => {
    const out = compiled(true);

    for (let i = 0; i < X.length; i++) expect(out[i]).toBe(X[i]);
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
      const wide = b.convert(sum.getResult(0), ScalarType.F64);
      b.returnOp([b.convert(wide.getResult(0), ScalarType.F32).getResult(0)]);
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
