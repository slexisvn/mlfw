import { describe, it, expect } from 'vitest';
import { tensor, sum, pow } from '../../src/index.js';

function gradValues(t) {
  return [...t.grad.data];
}

describe('pow backward', () => {
  it('d(x^2)/dx = 2x', () => {
    const x = tensor([3, 4], { requiresGrad: true });
    const n = tensor([2, 2]);
    sum(pow(x, n)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(6);
    expect(g[1]).toBeCloseTo(8);
  });

  it('d(x^3)/dx = 3x^2', () => {
    const x = tensor([2], { requiresGrad: true });
    const n = tensor([3]);
    pow(x, n).backward();
    expect(gradValues(x)[0]).toBeCloseTo(12);
  });

  it('d(x^n)/dn = x^n * ln(x)', () => {
    const x = tensor([2]);
    const n = tensor([3], { requiresGrad: true });
    pow(x, n).backward();
    expect(gradValues(n)[0]).toBeCloseTo(8 * Math.log(2));
  });

  it('d(x^0.5)/dx = 0.5 * x^(-0.5)', () => {
    const x = tensor([4], { requiresGrad: true });
    const half = tensor([0.5]);
    pow(x, half).backward();
    expect(gradValues(x)[0]).toBeCloseTo(0.5 * (1 / 2));
  });

  it('both x and n require grad simultaneously', () => {
    const x = tensor([Math.E], { requiresGrad: true });
    const n = tensor([2], { requiresGrad: true });
    pow(x, n).backward();
    expect(gradValues(x)[0]).toBeCloseTo(2 * Math.E);
    expect(gradValues(n)[0]).toBeCloseTo(Math.E ** 2 * Math.log(Math.E));
  });
});
