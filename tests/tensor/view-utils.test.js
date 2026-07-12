import { describe, it, expect } from 'vitest';
import {
  computeTranspose,
  computePermute,
  computeExpand,
  computeSlice,
  computeUnsqueeze,
  computeSqueeze,
  computeNarrow,
  computeSelect,
} from '../../src/tensor/utils/view_utils.js';

describe('computeTranspose', () => {
  it('swaps sizes and strides at given dims', () => {
    const r = computeTranspose([2, 3, 4], [12, 4, 1], 0, 2);
    expect(r.sizes).toEqual([4, 3, 2]);
    expect(r.strides).toEqual([1, 4, 12]);
  });

  it('handles negative dim indices', () => {
    const r = computeTranspose([2, 3], [3, 1], -2, -1);
    expect(r.sizes).toEqual([3, 2]);
    expect(r.strides).toEqual([1, 3]);
  });
});

describe('computePermute', () => {
  it('reorders sizes and strides according to dims', () => {
    const r = computePermute([2, 3, 4], [12, 4, 1], [2, 0, 1]);
    expect(r.sizes).toEqual([4, 2, 3]);
    expect(r.strides).toEqual([1, 12, 4]);
  });

  it('identity permutation returns same sizes/strides', () => {
    const r = computePermute([2, 3], [3, 1], [0, 1]);
    expect(r.sizes).toEqual([2, 3]);
    expect(r.strides).toEqual([3, 1]);
  });
});

describe('computeExpand', () => {
  it('broadcasts size-1 dims with stride 0', () => {
    const r = computeExpand([1, 3], [3, 1], [4, 3]);
    expect(r.sizes).toEqual([4, 3]);
    expect(r.strides).toEqual([0, 1]);
  });

  it('adds leading dims with stride 0', () => {
    const r = computeExpand([3], [1], [2, 3]);
    expect(r.sizes).toEqual([2, 3]);
    expect(r.strides).toEqual([0, 1]);
  });

  it('-1 in target preserves source size', () => {
    const r = computeExpand([2, 3], [3, 1], [-1, -1]);
    expect(r.sizes).toEqual([2, 3]);
    expect(r.strides).toEqual([3, 1]);
  });

  it('throws when source dim > 1 does not match target', () => {
    expect(() => computeExpand([2, 3], [3, 1], [4, 3])).toThrow(/Cannot expand/);
  });
});

describe('computeSlice', () => {
  it('computes correct sizes, strides, and offset', () => {
    const r = computeSlice([10], [1], 0, 2, 7, 1);
    expect(r.sizes).toEqual([5]);
    expect(r.strides).toEqual([1]);
    expect(r.offsetDelta).toBe(2);
  });

  it('handles step > 1', () => {
    const r = computeSlice([10], [1], 0, 0, 10, 3);
    expect(r.sizes).toEqual([4]);
    expect(r.strides).toEqual([3]);
    expect(r.offsetDelta).toBe(0);
  });

  it('handles negative start and end', () => {
    const r = computeSlice([10], [1], 0, -5, -1, 1);
    expect(r.sizes).toEqual([4]);
    expect(r.offsetDelta).toBe(5);
  });

  it('clamps out-of-bounds start and end', () => {
    const r = computeSlice([5], [1], 0, -100, 100, 1);
    expect(r.sizes).toEqual([5]);
    expect(r.offsetDelta).toBe(0);
  });

  it('returns empty when start >= end', () => {
    const r = computeSlice([10], [1], 0, 5, 3, 1);
    expect(r.sizes).toEqual([0]);
  });

  it('slices on non-zero dim of multidim tensor', () => {
    const r = computeSlice([3, 8], [8, 1], 1, 2, 6, 1);
    expect(r.sizes).toEqual([3, 4]);
    expect(r.strides).toEqual([8, 1]);
    expect(r.offsetDelta).toBe(2);
  });
});

describe('computeUnsqueeze', () => {
  it('inserts size-1 dim at start', () => {
    const r = computeUnsqueeze([3, 4], [4, 1], 0);
    expect(r.sizes).toEqual([1, 3, 4]);
    expect(r.strides[0]).toBe(12);
  });

  it('inserts size-1 dim at end', () => {
    const r = computeUnsqueeze([3, 4], [4, 1], 2);
    expect(r.sizes).toEqual([3, 4, 1]);
    expect(r.strides[2]).toBe(1);
  });

  it('handles negative dim (-1 inserts at end)', () => {
    const r = computeUnsqueeze([3, 4], [4, 1], -1);
    expect(r.sizes).toEqual([3, 4, 1]);
  });

  it('negative dim -2 inserts before last', () => {
    const r = computeUnsqueeze([3, 4], [4, 1], -2);
    expect(r.sizes).toEqual([3, 1, 4]);
  });
});

describe('computeSqueeze', () => {
  it('removes specified size-1 dim', () => {
    const r = computeSqueeze([1, 3, 1, 4], [12, 4, 4, 1], 0);
    expect(r.sizes).toEqual([3, 1, 4]);
  });

  it('does not remove dim if size != 1', () => {
    const r = computeSqueeze([2, 3], [3, 1], 0);
    expect(r.sizes).toEqual([2, 3]);
  });

  it('removes all size-1 dims when dim not specified', () => {
    const r = computeSqueeze([1, 3, 1, 4, 1], [12, 4, 4, 1, 1]);
    expect(r.sizes).toEqual([3, 4]);
  });

  it('handles all-ones shape when dim not specified', () => {
    const r = computeSqueeze([1, 1, 1], [1, 1, 1]);
    expect(r.sizes).toEqual([]);
  });
});

describe('computeNarrow', () => {
  it('narrows dim to specified range', () => {
    const r = computeNarrow([4, 5], [5, 1], 0, 1, 2);
    expect(r.sizes).toEqual([2, 5]);
    expect(r.offsetDelta).toBe(5);
  });

  it('handles negative dim', () => {
    const r = computeNarrow([4, 5], [5, 1], -1, 2, 3);
    expect(r.sizes).toEqual([4, 3]);
    expect(r.offsetDelta).toBe(2);
  });
});

describe('computeSelect', () => {
  it('removes dimension and computes offset', () => {
    const r = computeSelect([3, 4], [4, 1], 0, 1);
    expect(r.sizes).toEqual([4]);
    expect(r.strides).toEqual([1]);
    expect(r.offsetDelta).toBe(4);
  });

  it('handles negative index', () => {
    const r = computeSelect([3, 4], [4, 1], 0, -1);
    expect(r.sizes).toEqual([4]);
    expect(r.offsetDelta).toBe(8);
  });

  it('selects on last dim', () => {
    const r = computeSelect([3, 4], [4, 1], 1, 2);
    expect(r.sizes).toEqual([3]);
    expect(r.strides).toEqual([4]);
    expect(r.offsetDelta).toBe(2);
  });
});
