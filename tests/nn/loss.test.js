import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { mse_loss, binary_cross_entropy, cross_entropy } from '../../src/nn/functional/loss.js';
import { MSELoss, BCELoss, CrossEntropyLoss } from '../../src/nn/modules/loss.js';

describe('mse_loss', () => {
  it('computes mean squared error correctly', () => {
    const input = tensor([1, 2, 3]);
    const target = tensor([1, 2, 3]);
    const loss = mse_loss(input, target, 'mean');
    expect(loss.item()).toBeCloseTo(0);
  });

  it('computes correct value for known inputs', () => {
    const input = tensor([1, 2, 3]);
    const target = tensor([4, 5, 6]);
    const loss = mse_loss(input, target, 'mean');
    expect(loss.item()).toBeCloseTo(9);
  });

  it('sum reduction returns total squared error', () => {
    const input = tensor([0, 0]);
    const target = tensor([3, 4]);
    const loss = mse_loss(input, target, 'sum');
    expect(loss.item()).toBeCloseTo(25);
  });

  it('none reduction returns per-element loss', () => {
    const input = tensor([1, 2]);
    const target = tensor([3, 5]);
    const loss = mse_loss(input, target, 'none');
    const data = [...loss.data];
    expect(data[0]).toBeCloseTo(4);
    expect(data[1]).toBeCloseTo(9);
  });
});

describe('binary_cross_entropy', () => {
  it('returns near-zero loss for perfect predictions', () => {
    const input = tensor([0.999, 0.001]);
    const target = tensor([1, 0]);
    const loss = binary_cross_entropy(input, target, 'mean');
    expect(loss.item()).toBeLessThan(0.01);
  });

  it('returns high loss for completely wrong predictions', () => {
    const input = tensor([0.01, 0.99]);
    const target = tensor([1, 0]);
    const loss = binary_cross_entropy(input, target, 'mean');
    expect(loss.item()).toBeGreaterThan(2);
  });

  it('computes correct value for known inputs', () => {
    const p = 0.7;
    const input = tensor([p]);
    const target = tensor([1]);
    const expected = -(Math.log(p + 1e-7));
    const loss = binary_cross_entropy(input, target, 'mean');
    expect(loss.item()).toBeCloseTo(expected, 3);
  });
});

describe('MSELoss module', () => {
  it('produces same result as functional mse_loss', () => {
    const criterion = new MSELoss('mean');
    const input = tensor([1, 2, 3]);
    const target = tensor([2, 3, 4]);
    const loss = criterion.forward(input, target);
    expect(loss.item()).toBeCloseTo(1);
  });
});

describe('BCELoss module', () => {
  it('produces same result as functional bce', () => {
    const criterion = new BCELoss('mean');
    const input = tensor([0.5, 0.5]);
    const target = tensor([1, 0]);
    const loss = criterion.forward(input, target);
    const expected = -(Math.log(0.5 + 1e-7) + Math.log(0.5 + 1e-7)) / 2;
    expect(loss.item()).toBeCloseTo(expected, 3);
  });
});

describe('cross_entropy', () => {
  const logits = tensor([[2, 1, 0, 1], [0, 3, 1, 0], [1, 0, 2, 1], [0, 1, 0, 3]]);
  const target = tensor([0, 1, 2, 0], { dtype: 'i32' });

  it('is invariant to row order in the batch', () => {
    const ref = cross_entropy(logits, target, 'mean').item();
    const perm = [3, 1, 0, 2];
    const lp = tensor(perm.map(i => [...logits._impl.storage.data].slice(i * 4, i * 4 + 4)));
    const tp = tensor(perm.map(i => [...target._impl.storage.data][i]), { dtype: 'i32' });
    expect(cross_entropy(lp, tp, 'mean').item()).toBeCloseTo(ref, 6);
  });

  it('reads a strided (non-contiguous) target column correctly', () => {
    const ref = cross_entropy(logits, target, 'mean').item();
    const col = tensor([[0, 9], [1, 9], [2, 9], [0, 9]], { dtype: 'i32' }).select(1, 0);
    expect(col.isContiguous).toBe(false);
    expect(cross_entropy(logits, col, 'mean').item()).toBeCloseTo(ref, 6);
  });

  it('ignore_index excludes those targets and renormalizes the mean', () => {
    const loss = new CrossEntropyLoss('mean', 0);
    const tgt = tensor([0, 1, 0, 2], { dtype: 'i32' });
    const masked = loss.forward(logits, tgt).item();
    const onlyValid = cross_entropy(
      tensor([[0, 3, 1, 0], [0, 1, 0, 3]]),
      tensor([1, 2], { dtype: 'i32' }), 'mean',
    ).item();
    expect(masked).toBeCloseTo(onlyValid, 5);
  });

  it('default CrossEntropyLoss ignores nothing (back-compat)', () => {
    const loss = new CrossEntropyLoss();
    expect(loss.forward(logits, target).item()).toBeCloseTo(cross_entropy(logits, target, 'mean').item(), 6);
  });
});
