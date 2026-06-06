import { describe, it, expect } from 'vitest';
import {
  hasInternalOverlap,
  computeStorageRange,
  tensorsOverlap,
} from '../../src/tensor/utils/memory_overlap.js';
import { Storage } from '../../src/tensor/core/storage.js';
import { ScalarType } from '../../src/tensor/types/dtype.js';
import { CPU_DEVICE } from '../../src/tensor/types/device.js';

function makeOverlapImpl(sizes, strides, offset = 0, storage = null) {
  const s = storage || Storage.allocate(256, ScalarType.F32, CPU_DEVICE);
  return {
    sizes: () => sizes,
    strides: () => strides,
    storageOffset: () => offset,
    storage: () => s,
  };
}

describe('hasInternalOverlap', () => {
  it('scalar has no overlap', () => {
    expect(hasInternalOverlap(makeOverlapImpl([], []))).toBe(false);
  });

  it('contiguous 2D has no overlap', () => {
    expect(hasInternalOverlap(makeOverlapImpl([3, 4], [4, 1]))).toBe(false);
  });

  it('stride=0 with size>1 means broadcast overlap', () => {
    expect(hasInternalOverlap(makeOverlapImpl([4, 3], [0, 1]))).toBe(true);
  });

  it('stride=0 with size=1 is fine', () => {
    expect(hasInternalOverlap(makeOverlapImpl([1, 3], [0, 1]))).toBe(false);
  });

  it('cross-dim overlap when stride ratio < size', () => {
    expect(hasInternalOverlap(makeOverlapImpl([4, 4], [2, 1]))).toBe(true);
  });

  it('no cross-dim overlap when stride ratio >= size', () => {
    expect(hasInternalOverlap(makeOverlapImpl([3, 4], [4, 1]))).toBe(false);
  });

  it('transposed tensor has no overlap', () => {
    expect(hasInternalOverlap(makeOverlapImpl([4, 3], [1, 4]))).toBe(false);
  });

  it('1D always no overlap', () => {
    expect(hasInternalOverlap(makeOverlapImpl([10], [1]))).toBe(false);
    expect(hasInternalOverlap(makeOverlapImpl([10], [2]))).toBe(false);
  });
});

describe('computeStorageRange', () => {
  it('scalar at offset 0 spans [0, 1)', () => {
    const r = computeStorageRange(makeOverlapImpl([], [], 0));
    expect(r).toEqual({ start: 0, end: 1 });
  });

  it('scalar at offset 5 spans [5, 6)', () => {
    const r = computeStorageRange(makeOverlapImpl([], [], 5));
    expect(r).toEqual({ start: 5, end: 6 });
  });

  it('contiguous 1D [4] at offset 0', () => {
    const r = computeStorageRange(makeOverlapImpl([4], [1], 0));
    expect(r).toEqual({ start: 0, end: 4 });
  });

  it('contiguous 2D [2,3] at offset 0', () => {
    const r = computeStorageRange(makeOverlapImpl([2, 3], [3, 1], 0));
    expect(r).toEqual({ start: 0, end: 6 });
  });

  it('strided with offset', () => {
    const r = computeStorageRange(makeOverlapImpl([3], [2], 1));
    expect(r).toEqual({ start: 1, end: 6 });
  });

  it('negative stride shifts min offset down', () => {
    const r = computeStorageRange(makeOverlapImpl([4], [-1], 3));
    expect(r).toEqual({ start: 0, end: 4 });
  });

  it('zero-size dim returns empty range', () => {
    const r = computeStorageRange(makeOverlapImpl([0, 4], [4, 1], 0));
    expect(r).toEqual({ start: 0, end: 0 });
  });
});

describe('tensorsOverlap', () => {
  it('non-overlapping ranges return false', () => {
    const storage = Storage.allocate(64 * 4, ScalarType.F32, CPU_DEVICE);
    const a = makeOverlapImpl([4], [1], 0, storage);
    const b = makeOverlapImpl([4], [1], 10, storage);
    expect(tensorsOverlap(a, b)).toBe(false);
  });

  it('overlapping ranges return true', () => {
    const storage = Storage.allocate(64 * 4, ScalarType.F32, CPU_DEVICE);
    const a = makeOverlapImpl([8], [1], 0, storage);
    const b = makeOverlapImpl([8], [1], 4, storage);
    expect(tensorsOverlap(a, b)).toBe(true);
  });

  it('different storages never overlap', () => {
    const a = makeOverlapImpl([4], [1], 0);
    const b = makeOverlapImpl([4], [1], 0);
    expect(tensorsOverlap(a, b)).toBe(false);
  });

  it('adjacent ranges do not overlap', () => {
    const storage = Storage.allocate(64 * 4, ScalarType.F32, CPU_DEVICE);
    const a = makeOverlapImpl([4], [1], 0, storage);
    const b = makeOverlapImpl([4], [1], 4, storage);
    expect(tensorsOverlap(a, b)).toBe(false);
  });
});
