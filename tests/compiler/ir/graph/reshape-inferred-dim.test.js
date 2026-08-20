import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { TensorType, ScalarType, DYNAMIC, dimEquals } from '../../../../src/compiler/ir/graph/types.js';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';
import { SymInt } from '../../../../src/compiler/analysis/sym_int.js';

const reshape = registry.get('reshape');

function f32(shape) {
  return new TensorType(shape, ScalarType.F32);
}

function inferReshape(inputShape, newShape) {
  return reshape.inferResultTypes([f32(inputShape)], new Map([['new_shape', newShape]]))[0].shape;
}

function verifyReshape(inputShape, newShape) {
  const func = buildFunction('reshaped', [f32(inputShape)], [], (b, args) => {
    b.reshape(args[0], newShape);
    b.returnOp([]);
  });
  return verifyFunction(func).map(e => e.message);
}

describe('reshape resolves its inferred dimension', () => {
  const n = SymInt.var('n');

  it('divides the operand element count by the dimensions that are given', () => {
    expect(inferReshape([4, 3], [-1])).toEqual([12]);
    expect(inferReshape([2, 3, 4], [-1, 4])).toEqual([6, 4]);
    expect(inferReshape([2, 3, 4], [3, -1])).toEqual([3, 8]);
    expect(inferReshape([4, 3], [4, 3])).toEqual([4, 3]);
    expect(inferReshape([0, 3], [-1, 3])).toEqual([0, 3]);
  });

  it('yields DYNAMIC rather than a product of the dynamic sentinel', () => {
    expect(inferReshape([DYNAMIC, 3], [-1])).toEqual([DYNAMIC]);
    expect(inferReshape([DYNAMIC, 3], [-1, 3])).toEqual([DYNAMIC, 3]);
    expect(inferReshape([DYNAMIC, 2, 4], [-1, 8])).toEqual([DYNAMIC, 8]);
    expect(inferReshape([DYNAMIC, 3], [12])).toEqual([12]);
  });

  it('yields a SymInt expression when the operand carries symbolic dims', () => {
    const [flat] = inferReshape([n, 3], [-1]);
    expect(dimEquals(flat, SymInt.mul(SymInt.var('n'), 3))).toBe(true);
    expect(SymInt.evaluate(flat, new Map([['n', 4]]))).toBe(12);

    const kept = inferReshape([n, 3], [-1, 3]);
    expect(dimEquals(kept[0], n)).toBe(true);
    expect(kept[1]).toBe(3);

    const split = inferReshape([n, 6], [-1, 2, 3]);
    expect(dimEquals(split[0], n)).toBe(true);
    expect(SymInt.evaluate(split[0], new Map([['n', 5]]))).toBe(5);
  });

  it('leaves every other dimension untouched and never invents a negative extent', () => {
    for (const [inputShape, newShape] of [
      [[4, 3], [-1]], [[DYNAMIC, 3], [-1]], [[DYNAMIC, 3], [-1, 3]], [[n, 3], [-1]], [[4, 3], [2, -1, 3]],
    ]) {
      for (const d of inferReshape(inputShape, newShape)) {
        expect(typeof d === 'number' ? d >= 0 || d === DYNAMIC : d instanceof SymInt).toBe(true);
      }
    }
  });
});

describe('reshape verification', () => {
  it('accepts a reshape whose inferred dimension divides evenly', () => {
    expect(verifyReshape([4, 3], [-1])).toEqual([]);
    expect(verifyReshape([4, 3], [2, -1])).toEqual([]);
    expect(verifyReshape([DYNAMIC, 3], [-1])).toEqual([]);
  });

  it('does not second-guess a target shape it cannot evaluate', () => {
    expect(verifyReshape([12], [SymInt.var('n'), -1])).toEqual([]);
    expect(verifyReshape([12], [SymInt.var('n'), 4])).toEqual([]);
  });

  it('rejects an inferred dimension that does not divide the element count', () => {
    const errors = verifyReshape([4, 3], [-1, 5]);
    expect(errors.some(m => /cannot infer a dimension/.test(m))).toBe(true);
  });

  it('rejects more than one inferred dimension', () => {
    const errors = verifyReshape([4, 3], [-1, -1]);
    expect(errors.some(m => /at most one dynamic dimension/.test(m))).toBe(true);
  });

  it('still rejects a plain element-count mismatch', () => {
    const errors = verifyReshape([4, 3], [5, 3]);
    expect(errors.some(m => /numel mismatch: input 12 vs output 15/.test(m))).toBe(true);
  });
});

describe('negative extents other than DYNAMIC cannot cross a phase boundary', () => {
  it('flags a result type carrying a negative extent', () => {
    const func = buildFunction('leaked', [f32([4, 3])], [f32([12])], (b, args) => {
      const bad = b._buildOp('reshape', [args[0]], [f32([-3])], { new_shape: [-3] });
      b.returnOp([bad.getResult(0)]);
    });
    const errors = verifyFunction(func).map(e => e.message);
    expect(errors.some(m => /Result 0 dimension 0 is -3/.test(m))).toBe(true);
  });

  it('flags a declared signature carrying a negative extent', () => {
    const func = buildFunction('leaked_sig', [f32([-3])], [f32([-3])], (b, args) => {
      b.returnOp([args[0]]);
    });
    const errors = verifyFunction(func).map(e => e.message);
    expect(errors.some(m => /Input 0 dimension 0 is -3/.test(m))).toBe(true);
    expect(errors.some(m => /Output 0 dimension 0 is -3/.test(m))).toBe(true);
  });

  it('accepts DYNAMIC itself', () => {
    const func = buildFunction('dyn', [f32([DYNAMIC, 3])], [f32([DYNAMIC, 3])], (b, args) => {
      b.returnOp([args[0]]);
    });
    expect(verifyFunction(func).length).toBe(0);
  });
});
