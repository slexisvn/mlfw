import { describe, it, expect } from 'vitest';
import { tensor, add, mul, sum, ones } from '../../src/index.js';

function gradValues(t) {
  return [...t.grad.data];
}

function approx(arr, expected, tol = 4) {
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < arr.length; i++) {
    expect(arr[i]).toBeCloseTo(expected[i], tol);
  }
}

describe('backward with custom gradOutput', () => {
  it('uses provided gradient instead of default ones', () => {
    const a = tensor([1, 2, 3], { requiresGrad: true });
    const b = tensor([2, 2, 2]);
    const c = mul(a, b);
    c.backward(tensor([10, 20, 30]));
    approx(gradValues(a), [20, 40, 60]);
  });
});

describe('backward on non-scalar without gradOutput', () => {
  it('throws error for multi-element tensor', () => {
    const a = tensor([1, 2], { requiresGrad: true });
    const b = add(a, a);
    expect(() => b.backward()).toThrow(/non-scalar/);
  });
});

describe('backward on tensor without grad fn', () => {
  it('throws error', () => {
    expect(() => tensor([1, 2]).backward()).toThrow();
  });
});

describe('deep computation graph', () => {
  it('propagates gradient through 10 chained adds', () => {
    const x = tensor([1], { requiresGrad: true });
    let t = x;
    for (let i = 0; i < 10; i++) {
      t = add(t, x);
    }
    sum(t).backward();
    approx(gradValues(x), [11]);
  });
});

describe('wide fan-out graph', () => {
  it('accumulates gradient from multiple consumers', () => {
    const x = tensor([2], { requiresGrad: true });
    const a = mul(x, tensor([1]));
    const b = mul(x, tensor([2]));
    const c = mul(x, tensor([3]));
    sum(add(add(a, b), c)).backward();
    approx(gradValues(x), [6]);
  });
});

describe('diamond with different path weights', () => {
  it('sums gradients from both paths correctly', () => {
    const x = tensor([1], { requiresGrad: true });
    const left = mul(x, tensor([3]));
    const right = mul(x, tensor([7]));
    sum(mul(left, right)).backward();
    approx(gradValues(x), [3 * 7 + 7 * 3]);
  });
});

describe('releaseVariables after backward', () => {
  it('saved tensors are cleared after backward completes', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = tensor([3], { requiresGrad: true });
    const c = mul(a, b);
    const gradFn = c.gradFn;
    sum(c).backward();
    expect(gradFn.savedTensors().length).toBe(0);
  });
});
