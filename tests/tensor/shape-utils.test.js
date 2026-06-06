import { describe, it, expect } from 'vitest';
import {
  computeStrides,
  computeNumel,
  isContiguous,
  storageSize,
  broadcastShapes,
  broadcastShapesMulti,
  broadcastStrides,
  inferReshape,
  flatIndex,
  expandShape,
} from '../../src/tensor/utils/shape_utils.js';

describe('computeStrides', () => {
  it('computes row-major strides for 3D', () => {
    expect(computeStrides([3, 4, 5])).toEqual([20, 5, 1]);
  });

  it('returns empty array for scalar', () => {
    expect(computeStrides([])).toEqual([]);
  });

  it('returns [1] for 1D', () => {
    expect(computeStrides([7])).toEqual([1]);
  });

  it('handles 2D correctly', () => {
    expect(computeStrides([2, 6])).toEqual([6, 1]);
  });
});

describe('computeNumel', () => {
  it('multiplies all dimensions', () => {
    expect(computeNumel([2, 3, 4])).toBe(24);
  });

  it('returns 1 for scalar', () => {
    expect(computeNumel([])).toBe(1);
  });

  it('returns 0 when any dim is 0', () => {
    expect(computeNumel([0, 5])).toBe(0);
  });

  it('handles single dimension', () => {
    expect(computeNumel([10])).toBe(10);
  });
});

describe('isContiguous', () => {
  it('returns true for row-major strides', () => {
    expect(isContiguous([2, 6], [6, 1])).toBe(true);
  });

  it('returns false for transposed strides', () => {
    expect(isContiguous([2, 3], [1, 2])).toBe(false);
  });

  it('returns true for scalar', () => {
    expect(isContiguous([], [])).toBe(true);
  });

  it('treats size-1 dims as contiguous regardless of stride', () => {
    expect(isContiguous([1, 5], [999, 1])).toBe(true);
  });

  it('returns true when any dim is 0', () => {
    expect(isContiguous([0, 3], [3, 1])).toBe(true);
  });
});

describe('storageSize', () => {
  it('computes total storage needed', () => {
    expect(storageSize([2, 3], [3, 1], 0)).toBe(6);
  });

  it('accounts for offset', () => {
    expect(storageSize([2, 3], [3, 1], 5)).toBe(11);
  });

  it('returns offset+1 for scalar', () => {
    expect(storageSize([], [], 0)).toBe(1);
  });

  it('returns 0 when any dim is 0', () => {
    expect(storageSize([0, 3], [3, 1], 0)).toBe(0);
  });
});

describe('broadcastShapes', () => {
  it('broadcasts [3,1] with [1,4] to [3,4]', () => {
    expect(broadcastShapes([3, 1], [1, 4])).toEqual([3, 4]);
  });

  it('returns null for incompatible shapes', () => {
    expect(broadcastShapes([2, 3], [4, 3])).toBeNull();
  });

  it('broadcasts different ranks', () => {
    expect(broadcastShapes([5], [3, 5])).toEqual([3, 5]);
  });

  it('broadcasts identical shapes', () => {
    expect(broadcastShapes([2, 3], [2, 3])).toEqual([2, 3]);
  });

  it('broadcasts scalar with any shape', () => {
    expect(broadcastShapes([], [3, 4])).toEqual([3, 4]);
  });
});

describe('broadcastShapesMulti', () => {
  it('broadcasts 3 shapes', () => {
    expect(broadcastShapesMulti([[1, 3], [2, 1], [2, 3]])).toEqual([2, 3]);
  });

  it('returns null if any pair incompatible', () => {
    expect(broadcastShapesMulti([[2, 3], [4, 3]])).toBeNull();
  });

  it('returns empty for no shapes', () => {
    expect(broadcastShapesMulti([])).toEqual([]);
  });
});

describe('broadcastStrides', () => {
  it('sets stride to 0 for broadcast dimension', () => {
    const result = broadcastStrides([1, 3], [1, 1], [4, 3]);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(1);
  });

  it('prepends zeros for added dimensions', () => {
    const result = broadcastStrides([3], [1], [2, 3]);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(1);
  });
});

describe('inferReshape', () => {
  it('reshapes contiguous [2,3] to [6] without copy', () => {
    const r = inferReshape([2, 3], [3, 1], [6]);
    expect(r.sizes).toEqual([6]);
    expect(r.needsCopy).toBe(false);
  });

  it('infers -1 dimension', () => {
    const r = inferReshape([2, 3], [3, 1], [3, -1]);
    expect(r.sizes).toEqual([3, 2]);
    expect(r.needsCopy).toBe(false);
  });

  it('returns null for two -1 dims', () => {
    expect(inferReshape([6], [1], [-1, -1])).toBeNull();
  });

  it('returns null for numel mismatch', () => {
    expect(inferReshape([2, 3], [3, 1], [4, 2])).toBeNull();
  });

  it('sets needsCopy=true for non-contiguous reshape', () => {
    const r = inferReshape([2, 2], [3, 1], [4]);
    expect(r.needsCopy).toBe(true);
  });
});

describe('flatIndex', () => {
  it('computes flat index from multi-dim indices', () => {
    expect(flatIndex([1, 2], [3, 1], 0)).toBe(5);
  });

  it('accounts for offset', () => {
    expect(flatIndex([0, 0], [3, 1], 10)).toBe(10);
  });

  it('handles 3D', () => {
    expect(flatIndex([1, 0, 2], [12, 4, 1], 0)).toBe(14);
  });
});

describe('expandShape', () => {
  it('expands size-1 dims', () => {
    expect(expandShape([1, 3], [4, 3])).toEqual([4, 3]);
  });

  it('returns null for incompatible non-1 dim', () => {
    expect(expandShape([2, 3], [4, 3])).toBeNull();
  });

  it('returns null when source has more dims', () => {
    expect(expandShape([2, 3, 4], [3, 4])).toBeNull();
  });

  it('uses -1 to keep original size', () => {
    expect(expandShape([2, 3], [-1, 3])).toEqual([2, 3]);
  });

  it('prepends dimensions', () => {
    expect(expandShape([3], [2, 3])).toEqual([2, 3]);
  });

  it('expands multiple dims at once', () => {
    expect(expandShape([1, 1], [3, 5])).toEqual([3, 5]);
  });

  it('prepends multiple dims', () => {
    expect(expandShape([4], [2, 3, 4])).toEqual([2, 3, 4]);
  });
});

describe('inferReshape edge cases', () => {
  it('reshapes [1] to [] (scalar)', () => {
    const r = inferReshape([1], [1], []);
    expect(r).not.toBeNull();
    expect(r.sizes).toEqual([]);
  });

  it('reshapes [] to [1] (scalar to 1-element)', () => {
    const r = inferReshape([], [], [1]);
    expect(r).not.toBeNull();
    expect(r.sizes).toEqual([1]);
  });

  it('infers -1 with known dimensions', () => {
    const r = inferReshape([2, 3, 4], [12, 4, 1], [2, -1]);
    expect(r.sizes).toEqual([2, 12]);
  });

  it('returns null when -1 cannot divide evenly', () => {
    expect(inferReshape([7], [1], [2, -1])).toBeNull();
  });

  it('handles reshape with size-1 dims', () => {
    const r = inferReshape([6], [1], [1, 6, 1]);
    expect(r).not.toBeNull();
    expect(r.sizes).toEqual([1, 6, 1]);
  });
});

describe('broadcastShapes edge cases', () => {
  it('broadcasts scalar with scalar', () => {
    expect(broadcastShapes([], [])).toEqual([]);
  });

  it('broadcasts [1] with [1]', () => {
    expect(broadcastShapes([1], [1])).toEqual([1]);
  });

  it('broadcasts 3D with 1D', () => {
    expect(broadcastShapes([2, 3, 4], [4])).toEqual([2, 3, 4]);
  });

  it('returns null for [2] vs [3]', () => {
    expect(broadcastShapes([2], [3])).toBeNull();
  });
});

describe('isContiguous edge cases', () => {
  it('non-contiguous with gap between rows', () => {
    expect(isContiguous([2, 2], [3, 1])).toBe(false);
  });

  it('column-major 2D is not contiguous', () => {
    expect(isContiguous([3, 2], [1, 3])).toBe(false);
  });

  it('single element is always contiguous', () => {
    expect(isContiguous([1], [1])).toBe(true);
    expect(isContiguous([1, 1], [1, 1])).toBe(true);
  });
});

describe('computeStrides edge cases', () => {
  it('handles shape with zeros', () => {
    const s = computeStrides([2, 0, 3]);
    expect(s).toEqual([0, 3, 1]);
  });

  it('4D strides', () => {
    expect(computeStrides([2, 3, 4, 5])).toEqual([60, 20, 5, 1]);
  });
});
