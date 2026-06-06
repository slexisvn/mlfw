import { describe, it, expect } from 'vitest';
import { tensor, scalar, zeros, ones, fromBuffer, arange, eye, linspace, randn, full } from '../../src/index.js';
import { ScalarType } from '../../src/tensor/types/dtype.js';

describe('toArray on non-contiguous tensors', () => {
  it('sliced tensor produces correct nested array', () => {
    const t = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    const sliced = t.slice(1, 1, 3);
    expect(sliced.toArray()).toEqual([[2, 3], [6, 7], [10, 11]]);
  });

  it('transposed then sliced produces correct values', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]);
    const tr = t.transpose(0, 1);
    const s = tr.slice(0, 0, 2);
    expect(s.toArray()).toEqual([[1, 4], [2, 5]]);
  });

  it('permuted 3D tensor toArray round-trips', () => {
    const t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const p = t.permute([2, 0, 1]);
    expect(p.shape).toEqual([2, 2, 2]);
    const arr = p.toArray();
    expect(arr[0][0][0]).toBe(1);
    expect(arr[0][0][1]).toBe(3);
    expect(arr[0][1][0]).toBe(5);
    expect(arr[0][1][1]).toBe(7);
    expect(arr[1][0][0]).toBe(2);
    expect(arr[1][0][1]).toBe(4);
  });
});

describe('iterator on non-contiguous tensors', () => {
  it('iterates transposed rows correctly', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]);
    const tr = t.transpose(0, 1);
    const rows = [...tr];
    expect(rows.length).toBe(3);
    expect(rows[0].toArray()).toEqual([1, 4]);
    expect(rows[1].toArray()).toEqual([2, 5]);
    expect(rows[2].toArray()).toEqual([3, 6]);
  });

  it('iterates expanded tensor rows', () => {
    const t = tensor([[1, 2, 3]]);
    const expanded = t.expand([3, 3]);
    const rows = [...expanded];
    expect(rows.length).toBe(3);
    for (const row of rows) {
      expect(row.toArray()).toEqual([1, 2, 3]);
    }
  });
});

describe('_select edge cases', () => {
  it('select on 3D returns correct 2D slice', () => {
    const t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const selected = t._select(0, 1);
    expect(selected.shape).toEqual([2, 2]);
    expect(selected.toArray()).toEqual([[5, 6], [7, 8]]);
  });

  it('chained selects narrows to scalar', () => {
    const t = tensor([[10, 20, 30], [40, 50, 60]]);
    const val = t._select(0, 1)._select(0, 2);
    expect(val.shape).toEqual([]);
    expect(val.item()).toBe(60);
  });
});

describe('contiguity detection', () => {
  it('fresh tensor is contiguous', () => {
    expect(tensor([[1, 2], [3, 4]]).isContiguous).toBe(true);
  });

  it('transposed tensor is not contiguous', () => {
    expect(tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1).isContiguous).toBe(false);
  });

  it('expanded tensor is not contiguous', () => {
    expect(tensor([[1, 2, 3]]).expand([4, 3]).isContiguous).toBe(false);
  });

  it('sliced tensor with step>1 is not contiguous', () => {
    const t = arange(12).reshape([3, 4]);
    expect(t.slice(1, 0, 4, 2).isContiguous).toBe(false);
  });
});

describe('reshape view vs copy', () => {
  it('contiguous reshape shares storage', () => {
    const t = tensor([1, 2, 3, 4, 5, 6]);
    const r = t.reshape([2, 3]);
    r.data[0] = 999;
    expect(t.data[0]).toBe(999);
  });

  it('non-contiguous reshape that requires copy produces correct values', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1);
    const r = t.reshape([3, 2]);
    expect(r.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
  });
});

describe('arange overloads', () => {
  it('arange(n) produces [0..n-1]', () => {
    expect(arange(5).toArray()).toEqual([0, 1, 2, 3, 4]);
  });

  it('arange(start, end) with start > 0', () => {
    expect(arange(2, 5).toArray()).toEqual([2, 3, 4]);
  });

  it('arange(start, end, step) with step > 1', () => {
    expect(arange(0, 10, 3).toArray()).toEqual([0, 3, 6, 9]);
  });

  it('arange with fractional step', () => {
    const t = arange(0, 1, 0.25);
    expect(t.shape).toEqual([4]);
    expect(t.toArray()[0]).toBeCloseTo(0);
    expect(t.toArray()[3]).toBeCloseTo(0.75);
  });

  it('arange with start > end produces empty', () => {
    expect(arange(5, 2).shape).toEqual([0]);
  });

  it('arange with negative step', () => {
    expect(arange(5, 0, -1).toArray()).toEqual([5, 4, 3, 2, 1]);
  });
});

describe('eye rectangular', () => {
  it('eye(3, 5) has 1s on diagonal only', () => {
    const e = eye(3, 5);
    expect(e.shape).toEqual([3, 5]);
    const arr = e.toArray();
    expect(arr[0]).toEqual([1, 0, 0, 0, 0]);
    expect(arr[1]).toEqual([0, 1, 0, 0, 0]);
    expect(arr[2]).toEqual([0, 0, 1, 0, 0]);
  });

  it('eye(4, 2) truncates diagonal', () => {
    const e = eye(4, 2);
    const arr = e.toArray();
    expect(arr[0]).toEqual([1, 0]);
    expect(arr[1]).toEqual([0, 1]);
    expect(arr[2]).toEqual([0, 0]);
    expect(arr[3]).toEqual([0, 0]);
  });
});

describe('linspace', () => {
  it('linspace(0, 1, 5) produces evenly spaced values', () => {
    const t = linspace(0, 1, 5);
    expect(t.shape).toEqual([5]);
    expect(t.toArray()[0]).toBeCloseTo(0);
    expect(t.toArray()[2]).toBeCloseTo(0.5);
    expect(t.toArray()[4]).toBeCloseTo(1);
  });

  it('linspace with 1 step returns start', () => {
    const t = linspace(3, 7, 1);
    expect(t.toArray()).toEqual([3]);
  });

  it('linspace reverse direction', () => {
    const t = linspace(10, 0, 3);
    expect(t.toArray()[0]).toBeCloseTo(10);
    expect(t.toArray()[1]).toBeCloseTo(5);
    expect(t.toArray()[2]).toBeCloseTo(0);
  });
});

describe('randn statistical properties', () => {
  it('randn produces values with mean near 0', () => {
    const t = randn([10000]);
    const data = t.data;
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    const mean = sum / data.length;
    expect(Math.abs(mean)).toBeLessThan(0.1);
  });

  it('randn produces values with std near 1', () => {
    const t = randn([10000]);
    const data = t.data;
    let sum = 0, sum2 = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i];
      sum2 += data[i] * data[i];
    }
    const mean = sum / data.length;
    const variance = sum2 / data.length - mean * mean;
    expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.1);
  });

  it('randn with odd length fills all elements', () => {
    const t = randn([7]);
    expect(t.data.length).toBe(7);
    for (let i = 0; i < 7; i++) {
      expect(isFinite(t.data[i])).toBe(true);
    }
  });
});

describe('full', () => {
  it('full([2,3], 42) fills all with 42', () => {
    const t = full([2, 3], 42);
    expect(t.toArray()).toEqual([[42, 42, 42], [42, 42, 42]]);
  });
});

describe('fromBuffer with explicit shape', () => {
  it('creates tensor with correct shape from flat buffer', () => {
    const buf = new Float32Array([1, 2, 3, 4, 5, 6]);
    const t = fromBuffer(buf, [2, 3], ScalarType.F32);
    expect(t.shape).toEqual([2, 3]);
    expect(t.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  it('4D shape from flat buffer', () => {
    const buf = new Float32Array(24).fill(1);
    const t = fromBuffer(buf, [1, 2, 3, 4], ScalarType.F32);
    expect(t.shape).toEqual([1, 2, 3, 4]);
    expect(t.numel).toBe(24);
  });
});

describe('requiresGrad lifecycle', () => {
  it('default tensor does not require grad', () => {
    expect(tensor([1, 2, 3]).requiresGrad).toBe(false);
  });

  it('requiresGrad_ sets flag and updates dispatch keys', () => {
    const t = tensor([1, 2, 3]);
    t.requiresGrad_(true);
    expect(t.requiresGrad).toBe(true);
    expect(t.isLeaf).toBe(true);
  });

  it('detach removes grad tracking', () => {
    const t = tensor([1, 2, 3]);
    t.requiresGrad_(true);
    const d = t.detach();
    expect(d.requiresGrad).toBe(false);
  });
});

describe('tensor from nested array shape inference', () => {
  it('1D array', () => {
    expect(tensor([1, 2, 3]).shape).toEqual([3]);
  });

  it('2D array', () => {
    expect(tensor([[1, 2], [3, 4], [5, 6]]).shape).toEqual([3, 2]);
  });

  it('3D array', () => {
    expect(tensor([[[1], [2]], [[3], [4]]]).shape).toEqual([2, 2, 1]);
  });

  it('scalar number', () => {
    expect(scalar(5).shape).toEqual([]);
    expect(scalar(5).ndim).toBe(0);
  });
});
