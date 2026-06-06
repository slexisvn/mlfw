import { describe, it, expect } from 'vitest';
import {
  tensor, add, mul, sub, div, sum, neg, exp, noGrad,
} from '../../src/index.js';
import { GradMode } from '../../src/autograd/grad_mode.js';

function gradValues(t) {
  return [...t.grad.data];
}

function approx(arr, expected, tol = 4) {
  expect(arr.length).toBe(expected.length);
  for (let i = 0; i < arr.length; i++) {
    expect(arr[i]).toBeCloseTo(expected[i], tol);
  }
}

describe('autograd dispatch integration', () => {
  it('builds computation graph when requiresGrad is true', () => {
    const a = tensor([1, 2], { requiresGrad: true });
    const b = tensor([3, 4]);
    const c = add(a, b);
    expect(c.gradFn).not.toBeNull();
    expect(c.requiresGrad).toBe(true);
  });

  it('does not build graph when no input requires grad', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const c = add(a, b);
    expect(c.gradFn).toBeNull();
    expect(c.requiresGrad).toBe(false);
  });

  it('does not build graph when GradMode is disabled', () => {
    const a = tensor([1, 2], { requiresGrad: true });
    const c = noGrad(() => add(a, a));
    expect(c.gradFn).toBeNull();
  });

  it('chains multiple ops into connected graph', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = tensor([3], { requiresGrad: true });
    const c = mul(a, b);
    const d = add(c, a);
    sum(d).backward();
    expect([...a.grad.data]).toEqual([4]);
    expect([...b.grad.data]).toEqual([2]);
  });

  it('saves input tensors for backward computation', () => {
    const a = tensor([2, 3], { requiresGrad: true });
    const b = tensor([4, 5], { requiresGrad: true });
    const c = mul(a, b);

    const saved = c.gradFn.savedTensors();
    expect(saved.length).toBe(2);
    expect([...saved[0].data]).toEqual([2, 3]);
    expect([...saved[1].data]).toEqual([4, 5]);
  });

  it('connects gradFn edges to input accumulators', () => {
    const a = tensor([1], { requiresGrad: true });
    const b = tensor([2], { requiresGrad: true });
    const c = add(a, b);

    const edges = c.gradFn.nextEdges;
    expect(edges.length).toBe(2);
    expect(edges[0].node.name()).toBe('GradAccumulator');
    expect(edges[1].node.name()).toBe('GradAccumulator');
  });

  it('connects to previous gradFn when input is non-leaf', () => {
    const a = tensor([1], { requiresGrad: true });
    const b = add(a, a);
    const c = mul(b, b);

    const edges = c.gradFn.nextEdges;
    expect(edges[0].node).toBe(b.gradFn);
    expect(edges[1].node).toBe(b.gradFn);
  });

  it('diamond graph accumulates gradients from both paths', () => {
    const x = tensor([3], { requiresGrad: true });
    const left = mul(x, tensor([2]));
    const right = mul(x, tensor([5]));
    const out = add(left, right);
    sum(out).backward();
    approx(gradValues(x), [7]);
  });

  it('detach in mid-graph stops gradient on that branch only', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = tensor([3], { requiresGrad: true });
    const left = mul(a, b);
    const right = mul(a.detach(), b);
    sum(add(left, right)).backward();
    approx(gradValues(a), [3]);
    approx(gradValues(b), [2 + 2]);
  });

  it('long chain propagates gradient correctly: sum(exp(a*b + c))', () => {
    const a = tensor([1], { requiresGrad: true });
    const b = tensor([2], { requiresGrad: true });
    const c = tensor([0], { requiresGrad: true });
    sum(exp(add(mul(a, b), c))).backward();
    const e2 = Math.exp(2);
    approx(gradValues(a), [2 * e2]);
    approx(gradValues(b), [1 * e2]);
    approx(gradValues(c), [e2]);
  });

  it('variable used 3 times accumulates all gradient contributions', () => {
    const x = tensor([2], { requiresGrad: true });
    sum(add(add(x, x), x)).backward();
    approx(gradValues(x), [3]);
  });

  it('mixed requiresGrad: only grad-requiring inputs get gradients', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = tensor([5]);
    sum(mul(a, b)).backward();
    approx(gradValues(a), [5]);
    expect(b.grad).toBeNull();
  });

  it('neg(div(a, b)) computes correct gradients', () => {
    const a = tensor([6], { requiresGrad: true });
    const b = tensor([2], { requiresGrad: true });
    neg(div(a, b)).backward();
    approx(gradValues(a), [-1 / 2]);
    approx(gradValues(b), [6 / 4]);
  });

  it('sub in diamond: a - a = 0 but grad(a) = 0 not 2', () => {
    const a = tensor([5], { requiresGrad: true });
    sub(a, a).backward();
    approx(gradValues(a), [0]);
  });

  it('noGrad result used in grad-enabled context does not track', () => {
    const a = tensor([2], { requiresGrad: true });
    const b = noGrad(() => mul(a, tensor([3])));
    const c = add(a, b);
    sum(c).backward();
    approx(gradValues(a), [1]);
  });
});
