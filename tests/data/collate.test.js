import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { defaultCollate } from '../../src/data/collate.js';

describe('defaultCollate', () => {
  it('stacks tensors along new dim 0', () => {
    const batch = [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])];
    const result = defaultCollate(batch);
    expect(result.shape).toEqual([3, 2]);
    expect(result.toArray()).toEqual([[1, 2], [3, 4], [5, 6]]);
  });

  it('converts array of numbers to 1D tensor', () => {
    const result = defaultCollate([1, 2, 3, 4]);
    expect(result.shape).toEqual([4]);
    expect(result.toArray()).toEqual([1, 2, 3, 4]);
  });

  it('transposes and recurses arrays of arrays', () => {
    const batch = [
      [tensor([1]), tensor([10])],
      [tensor([2]), tensor([20])],
    ];
    const [col0, col1] = defaultCollate(batch);
    expect(col0.shape).toEqual([2, 1]);
    expect(col0.toArray()).toEqual([[1], [2]]);
    expect(col1.shape).toEqual([2, 1]);
    expect(col1.toArray()).toEqual([[10], [20]]);
  });

  it('recurses plain objects by key', () => {
    const batch = [
      { x: tensor([1, 2]), y: 10 },
      { x: tensor([3, 4]), y: 20 },
    ];
    const result = defaultCollate(batch);
    expect(result.x.shape).toEqual([2, 2]);
    expect(result.x.toArray()).toEqual([[1, 2], [3, 4]]);
    expect(result.y.shape).toEqual([2]);
    expect(result.y.toArray()).toEqual([10, 20]);
  });

  it('handles nested arrays of numbers', () => {
    const batch = [[1, 2], [3, 4]];
    const [col0, col1] = defaultCollate(batch);
    expect(col0.toArray()).toEqual([1, 3]);
    expect(col1.toArray()).toEqual([2, 4]);
  });

  it('throws on unsupported types', () => {
    expect(() => defaultCollate(['a', 'b'])).toThrow('unsupported element type');
  });

  it('handles single-element batch', () => {
    const result = defaultCollate([tensor([5, 6])]);
    expect(result.shape).toEqual([1, 2]);
    expect(result.toArray()).toEqual([[5, 6]]);
  });
});
