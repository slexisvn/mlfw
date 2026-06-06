import { describe, it, expect } from 'vitest';
import {
  tensor, sum, exp, log, sqrt, tanh, sigmoid, mul,
} from '../../src/index.js';

function gradValues(t) {
  return [...t.grad.data];
}

describe('exp backward', () => {
  it('d(exp(x))/dx = exp(x)', () => {
    const x = tensor([0, 1, 2], { requiresGrad: true });
    sum(exp(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(Math.exp(0));
    expect(g[1]).toBeCloseTo(Math.exp(1));
    expect(g[2]).toBeCloseTo(Math.exp(2));
  });
});

describe('log backward', () => {
  it('d(log(x))/dx = 1/x', () => {
    const x = tensor([1, 2, 4], { requiresGrad: true });
    sum(log(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(1);
    expect(g[1]).toBeCloseTo(0.5);
    expect(g[2]).toBeCloseTo(0.25);
  });
});

describe('sqrt backward', () => {
  it('d(sqrt(x))/dx = 1/(2*sqrt(x))', () => {
    const x = tensor([4, 9], { requiresGrad: true });
    sum(sqrt(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(1 / (2 * 2));
    expect(g[1]).toBeCloseTo(1 / (2 * 3));
  });
});

describe('tanh backward', () => {
  it('d(tanh(x))/dx = 1 - tanh(x)^2', () => {
    const x = tensor([0, 1], { requiresGrad: true });
    sum(tanh(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(1 - Math.tanh(0) ** 2);
    expect(g[1]).toBeCloseTo(1 - Math.tanh(1) ** 2);
  });
});

describe('sigmoid backward', () => {
  it('d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))', () => {
    const sig = v => 1 / (1 + Math.exp(-v));
    const x = tensor([0, 2], { requiresGrad: true });
    sum(sigmoid(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(sig(0) * (1 - sig(0)));
    expect(g[1]).toBeCloseTo(sig(2) * (1 - sig(2)));
  });
});

describe('composed unary backward', () => {
  it('d(exp(log(x)))/dx = 1 (identity for x>0)', () => {
    const x = tensor([2, 5], { requiresGrad: true });
    sum(exp(log(x))).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(1);
    expect(g[1]).toBeCloseTo(1);
  });

  it('d(sigmoid(2*x))/dx at x=0 is 0.5', () => {
    const x = tensor([0], { requiresGrad: true });
    const two = tensor([2]);
    sigmoid(mul(two, x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(0.5);
  });
});
