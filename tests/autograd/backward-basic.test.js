import { describe, it, expect } from 'vitest';
import {
  tensor, add, sub, mul, div, neg, sum, mean, noGrad,
} from '../../src/index.js';

function gradValues(t) {
  return [...t.grad.data];
}

function approx(arr, expected, tol = 4) {
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < arr.length; i++) {
    expect(arr[i]).toBeCloseTo(expected[i], tol);
  }
}

describe('neg backward', () => {
  it('negates the gradient', () => {
    const a = tensor([2, -3], { requiresGrad: true });
    sum(neg(a)).backward();
    approx(gradValues(a), [-1, -1]);
  });
});

describe('add backward', () => {
  it('propagates gradient equally to both inputs', () => {
    const a = tensor([1, 2, 3], { requiresGrad: true });
    const b = tensor([4, 5, 6], { requiresGrad: true });
    sum(add(a, b)).backward();
    approx(gradValues(a), [1, 1, 1]);
    approx(gradValues(b), [1, 1, 1]);
  });
});

describe('mul backward', () => {
  it('computes d(a*b)/da = b and d(a*b)/db = a', () => {
    const a = tensor([2, 3], { requiresGrad: true });
    const b = tensor([4, 5], { requiresGrad: true });
    sum(mul(a, b)).backward();
    approx(gradValues(a), [4, 5]);
    approx(gradValues(b), [2, 3]);
  });
});

describe('sub backward', () => {
  it('propagates positive grad to first, negative to second', () => {
    const a = tensor([1, 2], { requiresGrad: true });
    const b = tensor([3, 4], { requiresGrad: true });
    sum(sub(a, b)).backward();
    approx(gradValues(a), [1, 1]);
    approx(gradValues(b), [-1, -1]);
  });
});

describe('div backward', () => {
  it('computes d(a/b)/da = 1/b and d(a/b)/db = -a/b^2', () => {
    const a = tensor([6], { requiresGrad: true });
    const b = tensor([3], { requiresGrad: true });
    div(a, b).backward();
    approx(gradValues(a), [1 / 3]);
    approx(gradValues(b), [-6 / 9]);
  });
});

describe('chain rule', () => {
  it('propagates through a * b + c', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = tensor([3], { requiresGrad: true });
    const c = tensor([1], { requiresGrad: true });
    add(mul(a, b), c).backward();
    approx(gradValues(a), [3]);
    approx(gradValues(b), [2]);
    approx(gradValues(c), [1]);
  });

  it('propagates through (a+b)*a with shared variable', () => {
    const a = tensor([3], { requiresGrad: true });
    const b = tensor([2], { requiresGrad: true });
    mul(add(a, b), a).backward();
    approx(gradValues(a), [3 + (3 + 2)]);
    approx(gradValues(b), [3]);
  });
});

describe('sum backward', () => {
  it('expands scalar gradient back to input shape', () => {
    const a = tensor([1, 2, 3], { requiresGrad: true });
    sum(a).backward();
    approx(gradValues(a), [1, 1, 1]);
  });
});

describe('mean backward', () => {
  it('divides gradient by numel', () => {
    const a = tensor([1, 2, 3, 4], { requiresGrad: true });
    mean(a).backward();
    approx(gradValues(a), [0.25, 0.25, 0.25, 0.25]);
  });
});

describe('sum backward over axis', () => {
  it('expands gradient back along a single reduced axis (keepdim=false)', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    sum(sum(a, 1, false)).backward();
    approx(gradValues(a), [1, 1, 1, 1, 1, 1]);
  });

  it('handles keepdim=true', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    sum(sum(a, 1, true)).backward();
    approx(gradValues(a), [1, 1, 1, 1, 1, 1]);
  });
});

describe('mean backward over axis', () => {
  it('divides only by the reduced-axis size, not total numel', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    sum(mean(a, 1, false)).backward();
    approx(gradValues(a), [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]);
  });

  it('supports negative axis', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
    sum(mean(a, -1, false)).backward();
    approx(gradValues(a), [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]);
  });
});

describe('gradient accumulation', () => {
  it('sums gradients when a variable is used multiple times', () => {
    const a = tensor([2], { requiresGrad: true });
    add(a, a).backward();
    approx(gradValues(a), [2]);
  });
});

describe('noGrad prevents tracking', () => {
  it('operations inside noGrad do not build graph', () => {
    const a = tensor([1, 2], { requiresGrad: true });
    const b = noGrad(() => add(a, a));
    expect(b.requiresGrad).toBe(false);
    expect(b.gradFn).toBeNull();
  });
});

describe('detach stops gradient flow', () => {
  it('gradient does not flow through detached tensor', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = tensor([3], { requiresGrad: true });
    mul(a.detach(), b).backward();
    expect(a.grad).toBeNull();
    approx(gradValues(b), [2]);
  });
});

describe('backward error cases', () => {
  it('throws when calling backward on tensor without grad', () => {
    expect(() => tensor([1, 2]).backward()).toThrow();
  });
});
