import { describe, it, expect } from 'vitest';
import { tensor, arange } from '../../src/index.js';

describe('reshape', () => {
  it('reshapes 1D to 2D with correct values', () => {
    const t = tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  it('infers dimension with -1', () => {
    const t = tensor([1, 2, 3, 4]).reshape([2, -1]);
    expect(t.shape).toEqual([2, 2]);
  });

  it('throws on invalid reshape', () => {
    expect(() => tensor([1, 2, 3]).reshape([2, 2])).toThrow();
  });

  it('shares storage with original', () => {
    const t = tensor([1, 2, 3, 4]);
    const r = t.reshape([2, 2]);
    r.data[0] = 99;
    expect(t.data[0]).toBe(99);
  });

  it('flattens to 1D', () => {
    const t = tensor([[1, 2], [3, 4]]).reshape([-1]);
    expect(t.toArray()).toEqual([1, 2, 3, 4]);
  });

  it('reshapes non-contiguous (transposed) tensor via copy', () => {
    const tt = tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1);
    expect(tt.reshape([6]).toArray()).toEqual([1, 4, 2, 5, 3, 6]);
    expect(tt.reshape([2, 3]).toArray()).toEqual([[1, 4, 2], [5, 3, 6]]);
  });
});

describe('transpose', () => {
  it('transposes 2D tensor', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1);
    expect(t.shape).toEqual([3, 2]);
    expect(t.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it('shares storage', () => {
    const t = tensor([[1, 2], [3, 4]]);
    const tr = t.transpose(0, 1);
    expect(tr.data).toBe(t.data);
  });

  it('double transpose returns to original order', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]);
    const tt = t.transpose(0, 1).transpose(0, 1);
    expect(tt.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });
});

describe('permute', () => {
  it('permutes 3D tensor dimensions', () => {
    const t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const p = t.permute(2, 0, 1);
    expect(p.shape).toEqual([2, 2, 2]);
    expect(p.toArray()).toEqual([[[1, 3], [5, 7]], [[2, 4], [6, 8]]]);
  });
});

describe('expand', () => {
  it('broadcasts size-1 dimension', () => {
    const t = tensor([[1], [2], [3]]).expand([3, 4]);
    expect(t.shape).toEqual([3, 4]);
    expect(t.toArray()).toEqual([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
  });

  it('sets stride to 0 for expanded dim', () => {
    const t = tensor([[1], [2]]).expand([2, 3]);
    expect(t.strides[1]).toBe(0);
  });
});

describe('slice', () => {
  it('slices a range along dimension', () => {
    const t = arange(10);
    const s = t.slice(0, 2, 7);
    expect(s.shape).toEqual([5]);
    expect(s.toArray()).toEqual([2, 3, 4, 5, 6]);
  });

  it('slices 2D tensor along rows', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const s = t.slice(0, 0, 2);
    expect(s.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });
});

describe('unsqueeze', () => {
  it('adds dimension at position 0', () => {
    const t = tensor([1, 2, 3]).unsqueeze(0);
    expect(t.shape).toEqual([1, 3]);
    expect(t.toArray()).toEqual([[1, 2, 3]]);
  });

  it('adds dimension at the end', () => {
    const t = tensor([1, 2, 3]).unsqueeze(1);
    expect(t.shape).toEqual([3, 1]);
    expect(t.toArray()).toEqual([[1], [2], [3]]);
  });
});

describe('squeeze', () => {
  it('removes size-1 dimension', () => {
    const t = tensor([[1, 2, 3]]);
    expect(t.shape).toEqual([1, 3]);
    const s = t.squeeze(0);
    expect(s.shape).toEqual([3]);
    expect(s.toArray()).toEqual([1, 2, 3]);
  });

  it('does nothing when dim is not size 1', () => {
    const t = tensor([[1, 2], [3, 4]]);
    const s = t.squeeze(0);
    expect(s.shape).toEqual([2, 2]);
  });
});

describe('narrow', () => {
  it('narrows along dimension', () => {
    const t = arange(10);
    const n = t.narrow(0, 3, 4);
    expect(n.toArray()).toEqual([3, 4, 5, 6]);
  });

  it('narrows 2D tensor along columns', () => {
    const t = tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    const n = t.narrow(1, 1, 2);
    expect(n.toArray()).toEqual([[2, 3], [6, 7]]);
  });
});

describe('select', () => {
  it('selects along first dimension', () => {
    const t = tensor([[10, 20], [30, 40], [50, 60]]);
    const s = t.select(0, 1);
    expect(s.shape).toEqual([2]);
    expect(s.toArray()).toEqual([30, 40]);
  });

  it('selects along second dimension', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]);
    const s = t.select(1, 0);
    expect(s.shape).toEqual([2]);
    expect(s.toArray()).toEqual([1, 4]);
  });
});

describe('contiguous', () => {
  it('returns same tensor if already contiguous', () => {
    const t = tensor([1, 2, 3]);
    expect(t.contiguous()).toBe(t);
  });

  it('copies data for non-contiguous tensor', () => {
    const t = tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1);
    expect(t.isContiguous).toBe(false);
    const c = t.contiguous();
    expect(c.isContiguous).toBe(true);
    expect(c.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it('contiguous of a sliced (offset) view has exactly numel elements in .data', () => {
    const t = tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]).slice(1, 1, 5, 1);
    expect(t.shape).toEqual([2, 4]);
    const c = t.contiguous();
    expect(c.data.length).toBe(t.numel);
    expect(Array.from(c.data)).toEqual([11, 12, 13, 14, 21, 22, 23, 24]);
    expect(c.toArray()).toEqual([[11, 12, 13, 14], [21, 22, 23, 24]]);
  });
});

describe('t', () => {
  it('transposes 2D tensor', () => {
    const t = tensor([[1, 2], [3, 4]]).t();
    expect(t.toArray()).toEqual([[1, 3], [2, 4]]);
  });

  it('throws for non-2D tensor', () => {
    expect(() => tensor([[[1]]]).t()).toThrow(/2D/);
  });
});
