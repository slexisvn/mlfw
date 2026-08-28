import { describe, it, expect } from 'vitest';
import { tensor, scalar, zeros, ones, arange, empty } from '../../src/index.js';
import { META_DEVICE } from '../../src/tensor/types/device.js';

describe('item', () => {
  it('returns value of scalar tensor', () => {
    expect(scalar(5).item()).toBe(5);
  });

  it('throws for multi-element tensor', () => {
    expect(() => zeros([2, 3]).item()).toThrow(/exactly 1 element/);
  });
});

describe('toArray', () => {
  it('converts 2D tensor to nested array', () => {
    expect(tensor([[1, 2], [3, 4]]).toArray()).toEqual([[1, 2], [3, 4]]);
  });

  it('converts scalar to number', () => {
    expect(scalar(7).toArray()).toBe(7);
  });

  it('converts 3D tensor correctly', () => {
    const t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    expect(t.toArray()).toEqual([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
  });

  it('produces correct values for transposed tensor', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]);
    const transposed = t.transpose(0, 1);
    expect(transposed.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
  });
});

describe('Symbol.iterator', () => {
  it('iterates over first dimension with correct data', () => {
    const t = tensor([[1, 2], [3, 4]]);
    const rows = [...t];
    expect(rows[0].toArray()).toEqual([1, 2]);
    expect(rows[1].toArray()).toEqual([3, 4]);
  });

  it('throws for 0D tensor', () => {
    expect(() => [...scalar(1)]).toThrow(/0-d tensor/);
  });
});

describe('detach', () => {
  it('shares underlying data with original', () => {
    const t = tensor([1, 2, 3]);
    const d = t.detach();
    d.data[0] = 999;
    expect(t.data[0]).toBe(999);
  });
});

describe('_select', () => {
  it('selects correct row from 2D tensor', () => {
    const t = tensor([[10, 20], [30, 40], [50, 60]]);
    expect(t._select(0, 1).toArray()).toEqual([30, 40]);
  });

  it('selects correct column', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]);
    expect(t._select(1, 2).toArray()).toEqual([3, 6]);
  });
});

describe('data', () => {
  it('returns the whole buffer for a tensor that owns its storage', () => {
    const t = tensor([1, 2, 3, 4]);
    expect(t.data).toBe(t.storage.data);
    expect([...t.data]).toEqual([1, 2, 3, 4]);
  });

  it('returns only the elements of an offset view', () => {
    const v = arange(0, 8, 1).narrow(0, 4, 4);
    expect(v.data.length).toBe(v.numel);
    expect([...v.data]).toEqual(v.toArray());
    expect([...v.data]).toEqual([4, 5, 6, 7]);
  });

  it('returns only the elements of a truncated view', () => {
    const v = arange(0, 8, 1).narrow(0, 0, 3);
    expect([...v.data]).toEqual([0, 1, 2]);
  });

  it('sizes a multi-dimensional view by its element count, not its first axis', () => {
    const v = arange(0, 12, 1).reshape([3, 4]).narrow(0, 1, 2);
    expect(v.shape).toEqual([2, 4]);
    expect(v.data.length).toBe(8);
    expect([...v.data]).toEqual([4, 5, 6, 7, 8, 9, 10, 11]);
    expect(v.toArray()).toEqual([[4, 5, 6, 7], [8, 9, 10, 11]]);
  });

  it('aliases the parent storage so writes are visible through both', () => {
    const base = arange(0, 8, 1);
    const v = base.narrow(0, 4, 4);
    v.data[0] = 99;
    expect(base.toArray()[4]).toBe(99);
    expect(base.data.buffer).toBe(v.data.buffer);
  });

  it('refuses a non-contiguous tensor rather than returning the parent buffer', () => {
    const t = arange(0, 6, 1).reshape([2, 3]).transpose(0, 1);
    expect(t.isContiguous).toBe(false);
    expect(() => t.data).toThrow(/not contiguous/);
    expect([...t.contiguous().data]).toEqual([0, 3, 1, 4, 2, 5]);
  });

  it('returns null for a meta tensor', () => {
    expect(empty([2, 2], { device: META_DEVICE }).data).toBeNull();
  });
});
