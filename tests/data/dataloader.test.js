import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { TensorDataset } from '../../src/data/dataset.js';
import { SequentialSampler, BatchSampler } from '../../src/data/sampler.js';
import { DataLoader } from '../../src/data/dataloader.js';

function makeDataset(n) {
  const x = tensor(Array.from({ length: n }, (_, i) => [i, i * 10]));
  const y = tensor(Array.from({ length: n }, (_, i) => i * 100));
  return new TensorDataset(x, y);
}

describe('DataLoader sequential iteration', () => {
  it('iterates all data in order with default settings', () => {
    const ds = makeDataset(5);
    const loader = new DataLoader(ds, { batchSize: 2 });
    const batches = [...loader];
    expect(batches).toHaveLength(3);

    const [x0, y0] = batches[0];
    expect(x0.shape).toEqual([2, 2]);
    expect(x0.toArray()).toEqual([[0, 0], [1, 10]]);
    expect(y0.toArray()).toEqual([0, 100]);

    const [xLast, yLast] = batches[2];
    expect(xLast.shape).toEqual([1, 2]);
    expect(xLast.toArray()).toEqual([[4, 40]]);
    expect(yLast.toArray()).toEqual([400]);
  });

  it('supports dropLast', () => {
    const ds = makeDataset(5);
    const loader = new DataLoader(ds, { batchSize: 2, dropLast: true });
    const batches = [...loader];
    expect(batches).toHaveLength(2);
  });

  it('default batchSize is 1', () => {
    const ds = makeDataset(3);
    const loader = new DataLoader(ds);
    const batches = [...loader];
    expect(batches).toHaveLength(3);
    expect(batches[0][0].shape).toEqual([1, 2]);
  });
});

describe('DataLoader shuffled iteration', () => {
  it('covers all data when shuffle is true', () => {
    const ds = makeDataset(20);
    const loader = new DataLoader(ds, { batchSize: 4, shuffle: true });
    const batches = [...loader];
    expect(batches).toHaveLength(5);

    const allY = [];
    for (const [, y] of batches) {
      for (let i = 0; i < y.shape[0]; i++) {
        allY.push(y.data[i]);
      }
    }
    allY.sort((a, b) => a - b);
    const expected = Array.from({ length: 20 }, (_, i) => i * 100);
    expect(allY).toEqual(expected);
  });
});

describe('DataLoader custom sampler', () => {
  it('accepts a custom sampler', () => {
    const ds = makeDataset(6);
    const sampler = new SequentialSampler(ds);
    const loader = new DataLoader(ds, { sampler, batchSize: 3 });
    const batches = [...loader];
    expect(batches).toHaveLength(2);
    expect(batches[0][1].toArray()).toEqual([0, 100, 200]);
  });

  it('accepts a custom batchSampler', () => {
    const ds = makeDataset(6);
    const inner = new SequentialSampler(ds);
    const batchSampler = new BatchSampler(inner, 2);
    const loader = new DataLoader(ds, { batchSampler });
    const batches = [...loader];
    expect(batches).toHaveLength(3);
  });
});

describe('DataLoader custom collate', () => {
  it('uses a user-provided collate function', () => {
    const ds = makeDataset(4);
    const loader = new DataLoader(ds, {
      batchSize: 2,
      collate: (samples) => samples.length,
    });
    const batches = [...loader];
    expect(batches).toEqual([2, 2]);
  });
});

describe('DataLoader validation', () => {
  it('throws when batchSampler conflicts with batchSize', () => {
    const ds = makeDataset(4);
    const inner = new SequentialSampler(ds);
    const bs = new BatchSampler(inner, 2);
    expect(() => new DataLoader(ds, { batchSampler: bs, batchSize: 3 }))
      .toThrow('mutually exclusive');
  });

  it('throws when sampler and shuffle both provided', () => {
    const ds = makeDataset(4);
    const sampler = new SequentialSampler(ds);
    expect(() => new DataLoader(ds, { sampler, shuffle: true }))
      .toThrow('mutually exclusive');
  });
});

describe('DataLoader length', () => {
  it('returns correct number of batches', () => {
    const ds = makeDataset(10);
    expect(new DataLoader(ds, { batchSize: 3 }).length).toBe(4);
    expect(new DataLoader(ds, { batchSize: 3, dropLast: true }).length).toBe(3);
    expect(new DataLoader(ds, { batchSize: 5 }).length).toBe(2);
    expect(new DataLoader(ds, { batchSize: 10 }).length).toBe(1);
  });
});

describe('DataLoader re-iteration', () => {
  it('can iterate multiple times', () => {
    const ds = makeDataset(4);
    const loader = new DataLoader(ds, { batchSize: 2 });
    const run1 = [...loader];
    const run2 = [...loader];
    expect(run1).toHaveLength(2);
    expect(run2).toHaveLength(2);
    expect(run1[0][1].toArray()).toEqual(run2[0][1].toArray());
  });
});
