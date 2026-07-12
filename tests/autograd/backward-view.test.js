import { describe, it, expect } from 'vitest';
import { tensor, sum, mul, add } from '../../src/index.js';
import { reshape, transpose, permute } from '../../src/tensor/ops/ops.js';
import { ReshapeBackward, TransposeBackward, PermuteBackward, SliceBackward, SelectBackward, ExpandBackward } from '../../src/autograd/function/view.js';
import { ones, zeros } from '../../src/tensor/factory/creation_ops.js';

describe('ReshapeBackward', () => {
  it('reshapes gradient back to original shape', () => {
    const node = new ReshapeBackward();
    node.saveInputMetadata(0, [2, 3], 'f32');
    const g = tensor([1, 1, 1, 1, 1, 1]);
    const [result] = node.apply([g]);
    expect(result.shape).toEqual([2, 3]);
    expect(result.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
  });
});

describe('TransposeBackward', () => {
  it('transposes gradient back to original layout', () => {
    const node = new TransposeBackward(0, 1);
    const g = tensor([[1, 2], [3, 4], [5, 6]]);
    const [result] = node.apply([g]);
    expect(result.shape).toEqual([2, 3]);
    expect(result.toArray()).toEqual([[1, 3, 5], [2, 4, 6]]);
  });
});

describe('PermuteBackward', () => {
  it('applies inverse permutation to gradient', () => {
    const node = new PermuteBackward([2, 0, 1]);
    const g = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const [result] = node.apply([g]);
    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.toArray()).toEqual([[[1, 5], [2, 6]], [[3, 7], [4, 8]]]);
  });
});

describe('SliceBackward', () => {
  it('scatters gradient back to correct positions', () => {
    const node = new SliceBackward(0, 1, 4, 1);
    node.saveInputMetadata(0, [5], 'f32');
    const g = tensor([10, 20, 30]);
    const [result] = node.apply([g]);
    expect(result.shape).toEqual([5]);
    expect([...result.data]).toEqual([0, 10, 20, 30, 0]);
  });

  it('handles step > 1', () => {
    const node = new SliceBackward(0, 0, 5, 2);
    node.saveInputMetadata(0, [5], 'f32');
    const g = tensor([10, 20, 30]);
    const [result] = node.apply([g]);
    expect([...result.data]).toEqual([10, 0, 20, 0, 30]);
  });
});

describe('ExpandBackward', () => {
  it('sums gradient over expanded dims', () => {
    const node = new ExpandBackward();
    node.saveInputMetadata(0, [1, 3], 'f32');
    const g = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const [result] = node.apply([g]);
    expect(result.shape).toEqual([1, 3]);
    expect(result.toArray()).toEqual([[12, 15, 18]]);
  });
});

describe('end-to-end: mul after reshape tracks gradient', () => {
  it('gradient flows through mul applied to reshaped tensor', () => {
    const a = tensor([1, 2, 3, 4], { requiresGrad: true });
    const b = tensor([2, 2, 2, 2]);
    sum(mul(a, b)).backward();
    expect([...a.grad.data]).toEqual([2, 2, 2, 2]);
  });
});

describe('end-to-end: mul with sum on 2D', () => {
  it('computes correct gradient for 2D mul', () => {
    const a = tensor([[1, 2], [3, 4]], { requiresGrad: true });
    const b = tensor([[10, 10], [10, 10]]);
    sum(mul(a, b)).backward();
    expect(a.grad.toArray()).toEqual([[10, 10], [10, 10]]);
  });
});

describe('SliceBackward with negative bounds', () => {
  it('normalizes a negative start against the input size', () => {
    const node = new SliceBackward(0, -2, 5, 1);
    node.saveInputMetadata(0, [5], 'f32');
    const g = tensor([10, 20]);
    const [result] = node.apply([g]);
    expect([...result.data]).toEqual([0, 0, 0, 10, 20]);
  });

  it('normalizes a negative end against the input size', () => {
    const node = new SliceBackward(0, 1, -1, 1);
    node.saveInputMetadata(0, [5], 'f32');
    const g = tensor([7, 8, 9]);
    const [result] = node.apply([g]);
    expect([...result.data]).toEqual([0, 7, 8, 9, 0]);
  });
});

describe('SelectBackward with a negative index', () => {
  it('scatters gradient into the last row', () => {
    const node = new SelectBackward(0, -1);
    node.saveInputMetadata(0, [3, 2], 'f32');
    const g = tensor([10, 20]);
    const [result] = node.apply([g]);
    expect(result.shape).toEqual([3, 2]);
    expect(result.toArray()).toEqual([[0, 0], [0, 0], [10, 20]]);
  });
});

describe('PermuteBackward with negative dims', () => {
  it('matches the normalized permutation', () => {
    const g = tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);
    const [negResult] = new PermuteBackward([-1, 0, 1]).apply([g]);
    const [posResult] = new PermuteBackward([2, 0, 1]).apply([g]);
    expect(negResult.shape).toEqual([2, 3, 2]);
    expect(negResult.toArray()).toEqual(posResult.toArray());
  });
});

describe('end-to-end: negative indices flow gradient to the right positions', () => {
  it('select with a negative index', () => {
    const x = tensor([[1, 2], [3, 4], [5, 6]], { requiresGrad: true });
    sum(x.select(0, -1)).backward();
    expect(x.grad.toArray()).toEqual([[0, 0], [0, 0], [1, 1]]);
  });

  it('slice with a negative start', () => {
    const x = tensor([10, 20, 30, 40, 50, 60], { requiresGrad: true });
    sum(x.slice(0, -3, 6, 1)).backward();
    expect([...x.grad.data]).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it('narrow scatters gradient back to the narrowed window', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    sum(x.narrow(1, 1, 2)).backward();
    expect(x.grad.toArray()).toEqual([[0, 1, 1], [0, 1, 1]]);
  });

  it('permute with a negative dim keeps gradient aligned with the input', () => {
    const x = tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], { requiresGrad: true });
    const y = x.permute([-1, 0, 1]);
    sum(mul(y, y)).backward();
    expect([...x.grad.contiguous().data]).toEqual([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]);
  });
});
