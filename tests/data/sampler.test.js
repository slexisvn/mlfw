import { describe, it, expect } from 'vitest';
import {
  SequentialSampler,
  RandomSampler,
  BatchSampler,
} from '../../src/data/sampler.js';

function mockDataSource(n) {
  return { length: n };
}

describe('SequentialSampler', () => {
  it('yields indices 0 through n-1', () => {
    const sampler = new SequentialSampler(mockDataSource(5));
    expect([...sampler]).toEqual([0, 1, 2, 3, 4]);
  });

  it('yields nothing for empty source', () => {
    const sampler = new SequentialSampler(mockDataSource(0));
    expect([...sampler]).toEqual([]);
  });

  it('produces fresh iterator each time', () => {
    const sampler = new SequentialSampler(mockDataSource(3));
    expect([...sampler]).toEqual([0, 1, 2]);
    expect([...sampler]).toEqual([0, 1, 2]);
  });
});

describe('RandomSampler', () => {
  it('yields all indices exactly once', () => {
    const sampler = new RandomSampler(mockDataSource(100));
    const indices = [...sampler];
    expect(indices).toHaveLength(100);
    expect(new Set(indices).size).toBe(100);
    for (let i = 0; i < 100; i++) {
      expect(indices).toContain(i);
    }
  });

  it('produces different orderings on separate iterations', () => {
    const sampler = new RandomSampler(mockDataSource(50));
    const run1 = [...sampler];
    const run2 = [...sampler];
    expect(run1).toHaveLength(50);
    expect(run2).toHaveLength(50);
    const same = run1.every((v, i) => v === run2[i]);
    expect(same).toBe(false);
  });

  it('handles single-element source', () => {
    const sampler = new RandomSampler(mockDataSource(1));
    expect([...sampler]).toEqual([0]);
  });

  it('handles empty source', () => {
    const sampler = new RandomSampler(mockDataSource(0));
    expect([...sampler]).toEqual([]);
  });
});

describe('BatchSampler', () => {
  it('groups indices into batches of batchSize', () => {
    const inner = new SequentialSampler(mockDataSource(10));
    const batched = new BatchSampler(inner, 3);
    const batches = [...batched];
    expect(batches).toEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]);
  });

  it('drops last incomplete batch when dropLast is true', () => {
    const inner = new SequentialSampler(mockDataSource(10));
    const batched = new BatchSampler(inner, 3, true);
    const batches = [...batched];
    expect(batches).toEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
  });

  it('yields nothing for empty sampler', () => {
    const inner = new SequentialSampler(mockDataSource(0));
    const batched = new BatchSampler(inner, 4);
    expect([...batched]).toEqual([]);
  });

  it('handles exact divisible case', () => {
    const inner = new SequentialSampler(mockDataSource(6));
    const batched = new BatchSampler(inner, 3);
    expect([...batched]).toEqual([[0, 1, 2], [3, 4, 5]]);
  });

  it('composes with RandomSampler', () => {
    const inner = new RandomSampler(mockDataSource(10));
    const batched = new BatchSampler(inner, 5);
    const batches = [...batched];
    expect(batches).toHaveLength(2);
    const allIndices = batches.flat();
    expect(new Set(allIndices).size).toBe(10);
  });

  it('batchSize=1 yields single-element arrays', () => {
    const inner = new SequentialSampler(mockDataSource(3));
    const batched = new BatchSampler(inner, 1);
    expect([...batched]).toEqual([[0], [1], [2]]);
  });
});
