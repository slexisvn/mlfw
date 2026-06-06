import { describe, it, expect } from 'vitest';
import { tensor, scalar, zeros, ones } from '../../src/index.js';

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
