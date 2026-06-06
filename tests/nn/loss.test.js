import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { mse_loss, binary_cross_entropy } from '../../src/nn/functional/loss.js';
import { MSELoss, BCELoss } from '../../src/nn/modules/loss.js';

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
