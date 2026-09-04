import { describe, it, expect } from 'vitest';
import { TensorType, ScalarType, Layout, DYNAMIC, broadcastDim, dimEquals } from '../../../../src/compiler/ir/graph/types.js';
import { SymInt } from '../../../../src/compiler/ir/sym_int.js';
import { registry } from '../../../../src/compiler/ir/graph/ops.js';

import { GraphFunction } from '../../../../src/compiler/ir/graph/function.js';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { learnDynamicExtents, learnSymbolNames } from '../../../../src/compiler/ir/graph/symbolic_shape.js';
import { unifyShapeSymbols } from '../../../../src/compiler/ir/graph/shape_symbols.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { RuntimeTensor } from '../../../../src/runtime/runtime.js';

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

describe('symbolic shape learning', () => {
  const value = (shape) => new GraphFunction('f', [new TensorType(shape, 'f32')], []).args[0];

  it('learns only unknown dynamic extents and preserves static dimensions', () => {
    const v = value([DYNAMIC, 4, DYNAMIC]);
    expect(learnDynamicExtents(v, ['batch', 99, DYNAMIC])).toBe(true);
    expect(v.symbolicShape).toEqual(['batch', 4, DYNAMIC]);
    expect(learnDynamicExtents(v, ['other', 99, 8])).toBe(true);
    expect(v.symbolicShape).toEqual(['batch', 4, 8]);
    expect(learnDynamicExtents(v, ['other', 99, 9])).toBe(false);
    expect(v.symbolicShape).toEqual(['batch', 4, 8]);
  });

  it('learns names over numeric samples but never replaces a known name', () => {
    const v = value([DYNAMIC, DYNAMIC]);
    v.symbolicShape = [3, 'width'];
    expect(learnSymbolNames(v, ['batch', 'other'])).toBe(true);
    expect(v.symbolicShape).toEqual(['batch', 'width']);
    expect(learnSymbolNames(v, [5, 6])).toBe(false);
  });

  it('ignores rank mismatches without assigning shape metadata', () => {
    const v = value([DYNAMIC, 4]);
    expect(learnDynamicExtents(v, ['batch'])).toBe(false);
    expect(v.symbolicShape).toBeUndefined();
  });

  it('propagates names backward and forward across an elementwise chain to a fixed point', () => {
    const t = new TensorType([DYNAMIC, 4], 'f32');
    const func = buildFunction('chain', [t, t], [t], (b, [x, y]) => {
      const neg = b.neg(x).getResult(0);
      const add = b.add(neg, y).getResult(0);
      b.returnOp([b.tanh(add).getResult(0)]);
    });
    func.args[1].symbolicShape = ['batch', 4];
    unifyShapeSymbols(func);
    for (const v of [...func.args, ...func.opsArray().flatMap((op) => op.results)]) {
      expect(v.symbolicShape).toEqual(['batch', 4]);
    }
    unifyShapeSymbols(func);
    expect(func.args[0].symbolicShape).toEqual(['batch', 4]);
    const compiled = compileGraph(func, CPUTarget());
    for (const rows of [2, 5]) {
      const x = Float32Array.from({ length: rows * 4 }, (_, i) => i / 8);
      const y = Float32Array.from({ length: rows * 4 }, (_, i) => (i % 3) / 4);
      const out = new Float32Array(rows * 4).fill(NaN);
      compiled.run('chain', ...[x, y, out].map((data) => new RuntimeTensor(data, [rows, 4], 'f32')));
      for (let i = 0; i < out.length; i++) expect(out[i], 'row count ' + rows + ', element ' + i).toBeCloseTo(Math.tanh(y[i] - x[i]), 6);
    }
  });

  it('does not copy result symbols through an unrelated reshape', () => {
    const input = new TensorType([DYNAMIC, 2], 'f32');
    const output = new TensorType([2, DYNAMIC], 'f32');
    const func = buildFunction('reshape', [input], [output], (b, [x]) => {
      const reshape = b.reshape(x, output.shape);
      reshape.getResult(0).symbolicShape = [2, 'batch'];
      b.returnOp(reshape.results);
    });
    unifyShapeSymbols(func);
    expect(func.args[0].symbolicShape).toBeUndefined();
  });
});

describe('removing tensor axes', () => {
  it.each([
    [[2, 0, 1], 0, [1, 0]],
    [[2, 0, 1], 1, [1, 0]],
    [[2, 0, 1], 2, [0, 1]],
    [[0], 0, []],
  ])('drops an axis from layout %j at index %i', (order, axis, expected) => {
    const layout = new Layout(order);
    expect(layout.dropDim(axis).order).toEqual(expected);
    expect(layout.order).toEqual(order);
  });

  it('drops the leading extent while retaining dtype and remaining layout order', () => {
    const type = new TensorType([3, DYNAMIC, 5], 'i32', new Layout([2, 0, 1]));
    const step = type.dropLeadingAxis();
    expect(step.shape).toEqual([DYNAMIC, 5]);
    expect(step.layout.order).toEqual([1, 0]);
    expect(step.dtype).toBe('i32');
    expect(type.shape).toEqual([3, DYNAMIC, 5]);
  });
});
