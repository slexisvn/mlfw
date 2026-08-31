import { describe, it, expect } from 'vitest';
import { TensorType, ScalarType, DYNAMIC, broadcastDim, dimEquals } from '../../../../src/compiler/ir/graph/types.js';
import { SymInt } from '../../../../src/compiler/ir/sym_int.js';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';

describe('symbolic dynamic shapes', () => {
  const n = SymInt.var('n');
  const m = SymInt.var('m');
  const k = SymInt.var('k');

  it('TensorType holds symbolic dims: hasDynamic + isFullyStatic', () => {
    const t = new TensorType([n, 4], ScalarType.F32);
    expect(t.hasDynamic).toBe(true);
    expect(t.isFullyStatic).toBe(false);
    expect(new TensorType([3, 4], ScalarType.F32).isFullyStatic).toBe(true);
    expect(new TensorType([3, 4], ScalarType.F32).hasDynamic).toBe(false);
    expect(new TensorType([DYNAMIC, 4], ScalarType.F32).hasDynamic).toBe(true);
  });

  it('dimEquals + shapeEquals reason structurally over symbolic dims', () => {
    expect(dimEquals(n, SymInt.var('n'))).toBe(true);
    expect(dimEquals(n, m)).toBe(false);
    expect(dimEquals(SymInt.mul(n, m), SymInt.mul(SymInt.var('n'), SymInt.var('m')))).toBe(true);
    const a = new TensorType([n, 4], ScalarType.F32);
    expect(a.shapeEquals(new TensorType([SymInt.var('n'), 4], ScalarType.F32))).toBe(true);
    expect(a.shapeEquals(new TensorType([m, 4], ScalarType.F32))).toBe(false);
    expect(a.equals(new TensorType([SymInt.var('n'), 4], ScalarType.F32))).toBe(true);
  });

  it('broadcastDim preserves symbolic dims and is conservative on unprovable pairs', () => {
    expect(dimEquals(broadcastDim(n, 1), n)).toBe(true);
    expect(dimEquals(broadcastDim(1, n), n)).toBe(true);
    expect(dimEquals(broadcastDim(n, SymInt.var('n')), n)).toBe(true);
    expect(broadcastDim(n, m)).toBe(DYNAMIC);
    expect(broadcastDim(2, 3)).toBe(null);
  });

  it('static shapes are unaffected (regression guard)', () => {
    const a = new TensorType([3, 4], ScalarType.F32);
    expect(a.equals(new TensorType([3, 4], ScalarType.F32))).toBe(true);
    expect(a.shapeEquals(new TensorType([3, 5], ScalarType.F32))).toBe(false);
    expect(TensorType.broadcastShape([3, 1], [3, 4])).toEqual([3, 4]);
    expect(a.numel()).toBe(12);
  });

  it('elementwise add infers symbolic broadcast shape', () => {
    const add = registry.get('add');
    const t = new TensorType([n, 4], ScalarType.F32);
    const [res] = add.inferResultTypes([t, t]);
    expect(res.shapeEquals(t)).toBe(true);
    const [res2] = add.inferResultTypes([new TensorType([n, 1], ScalarType.F32), new TensorType([n, 4], ScalarType.F32)]);
    expect(res2.shapeEquals(t)).toBe(true);
  });

  it('transpose permutes symbolic dims', () => {
    const tr = registry.get('transpose');
    const [res] = tr.inferResultTypes([new TensorType([n, m], ScalarType.F32)], new Map([['permutation', [1, 0]]]));
    expect(dimEquals(res.shape[0], m)).toBe(true);
    expect(dimEquals(res.shape[1], n)).toBe(true);
  });

  it('dot propagates symbolic free dims; verify accepts symbolic contracting agreement', () => {
    const dot = registry.get('dot');
    const lhs = new TensorType([n, k], ScalarType.F32);
    const rhs = new TensorType([k, m], ScalarType.F32);
    const attrs = new Map([['lhs_contracting', [1]], ['rhs_contracting', [0]]]);
    const [res] = dot.inferResultTypes([lhs, rhs], attrs);
    expect(res.shape.length).toBe(2);
    expect(dimEquals(res.shape[0], n)).toBe(true);
    expect(dimEquals(res.shape[1], m)).toBe(true);

    const mockOp = {
      numOperands: 2,
      getOperand: (i) => ({ type: i === 0 ? lhs : rhs }),
      hasAttr: (key) => key === 'lhs_contracting' || key === 'rhs_contracting',
      getAttr: (key) => ({ lhs_contracting: [1], rhs_contracting: [0] }[key]),
    };
    expect(dot.verify(mockOp)).toEqual([]);

    const bad = {
      numOperands: 2,
      getOperand: (i) => ({ type: i === 0 ? new TensorType([n, 8], ScalarType.F32) : new TensorType([4, m], ScalarType.F32) }),
      hasAttr: (key) => key === 'lhs_contracting' || key === 'rhs_contracting',
      getAttr: (key) => ({ lhs_contracting: [1], rhs_contracting: [0] }[key]),
    };
    expect(dot.verify(bad).length).toBeGreaterThan(0);
  });
});
