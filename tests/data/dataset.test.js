import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { Dataset, TensorDataset, MapDataset } from '../../src/data/dataset.js';

describe('Dataset base class', () => {
  it('throws on unimplemented length', () => {
    const ds = new Dataset();
    expect(() => ds.length).toThrow('Subclass must implement');
  });

  it('throws on unimplemented get', () => {
    const ds = new Dataset();
    expect(() => ds.get(0)).toThrow('Subclass must implement');
  });
});

describe('TensorDataset', () => {
  it('returns correct length from dim 0', () => {
    const x = tensor([[1, 2], [3, 4], [5, 6]]);
    const ds = new TensorDataset(x);
    expect(ds.length).toBe(3);
  });

  it('get returns zero-copy view via select', () => {
    const x = tensor([[10, 20], [30, 40]]);
    const ds = new TensorDataset(x);
    const sample = ds.get(1);
    expect(sample).toHaveLength(1);
    expect(sample[0].toArray()).toEqual([30, 40]);
  });

  it('supports multiple tensors with matching dim 0', () => {
    const x = tensor([[1, 2], [3, 4], [5, 6]]);
    const y = tensor([10, 20, 30]);
    const ds = new TensorDataset(x, y);
    expect(ds.length).toBe(3);

    const [xi, yi] = ds.get(2);
    expect(xi.toArray()).toEqual([5, 6]);
    expect(yi.item()).toBe(30);
  });

  it('throws on dim 0 mismatch', () => {
    const x = tensor([[1, 2], [3, 4]]);
    const y = tensor([1, 2, 3]);
    expect(() => new TensorDataset(x, y)).toThrow('Size mismatch');
  });

  it('throws on empty constructor', () => {
    expect(() => new TensorDataset()).toThrow('at least one tensor');
  });

  it('iterates all samples via Symbol.iterator', () => {
    const x = tensor([[1], [2], [3]]);
    const ds = new TensorDataset(x);
    const items = [...ds];
    expect(items).toHaveLength(3);
    expect(items[0][0].item()).toBe(1);
    expect(items[2][0].item()).toBe(3);
  });

  it('get returns views sharing the same underlying storage', () => {
    const x = tensor([[1, 2], [3, 4]]);
    const ds = new TensorDataset(x);
    const view = ds.get(0)[0];
    expect(view.data.buffer).toBe(x.data.buffer);
  });
});

describe('MapDataset', () => {
  it('applies transform lazily on get', () => {
    const x = tensor([[1, 2], [3, 4]]);
    const base = new TensorDataset(x);
    const mapped = new MapDataset(base, (sample) => {
      return sample.map(t => t.toArray().map(v => v * 2));
    });
    expect(mapped.length).toBe(2);
    expect(mapped.get(0)).toEqual([[2, 4]]);
    expect(mapped.get(1)).toEqual([[6, 8]]);
  });

  it('composes with another MapDataset', () => {
    const x = tensor([1, 2, 3]);
    const base = new TensorDataset(x);
    const m1 = new MapDataset(base, (s) => s[0].item());
    const m2 = new MapDataset(m1, (v) => v * 10);
    expect(m2.get(0)).toBe(10);
    expect(m2.get(2)).toBe(30);
  });

  it('iterates via inherited Symbol.iterator', () => {
    const x = tensor([10, 20, 30]);
    const base = new TensorDataset(x);
    const mapped = new MapDataset(base, (s) => s[0].item() + 1);
    expect([...mapped]).toEqual([11, 21, 31]);
  });
});
