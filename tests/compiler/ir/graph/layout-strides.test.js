import { describe, it, expect } from 'vitest';
import { Layout, TensorType, ScalarType, DYNAMIC } from '../../../../src/compiler/ir/graph/types.js';
import { SymInt } from '../../../../src/compiler/ir/sym_int.js';

describe('Layout.computeStrides', () => {
  it('lays out a row-major tensor with the last axis contiguous', () => {
    expect(Layout.rowMajor(3).computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
  });

  it('lays out a column-major tensor with the first axis contiguous', () => {
    expect(Layout.columnMajor(3).computeStrides([2, 3, 4])).toEqual([1, 2, 6]);
  });

  it('indexes the result by logical dimension, not by storage position', () => {
    expect(new Layout([1, 0]).computeStrides([4, 8])).toEqual([1, 4]);
  });

  it('marks a stride dynamic once a faster-varying axis is dynamic', () => {
    expect(Layout.rowMajor(3).computeStrides([2, DYNAMIC, 4])).toEqual([DYNAMIC, 4, 1]);
  });

  it('marks a stride dynamic once a faster-varying axis is symbolic', () => {
    expect(Layout.rowMajor(3).computeStrides([2, SymInt.var('n'), 4])).toEqual([DYNAMIC, 4, 1]);
  });

  it('returns one stride per shape dimension even when the layout has a different rank', () => {
    const strides = Layout.rowMajor(2).computeStrides([5, 2, 3]);
    expect(strides.length).toBe(3);
    expect(strides.every(Number.isInteger)).toBe(true);
  });
});

describe('TensorType.withShape', () => {
  it('keeps a matching layout when the rank is unchanged', () => {
    const type = new TensorType([4, 8], ScalarType.F32, new Layout([1, 0]));
    expect(type.withShape([2, 16]).layout.order).toEqual([1, 0]);
  });

  it('gives up a layout that no longer describes the new rank', () => {
    const type = new TensorType([4, 8], ScalarType.F32, new Layout([1, 0]));
    const grown = type.withShape([3, 4, 8]);
    expect(grown.layout.rank).toBe(3);
    expect(grown.layout.isIdentity()).toBe(true);
    expect(grown.strides()).toEqual([32, 8, 1]);
  });
});
