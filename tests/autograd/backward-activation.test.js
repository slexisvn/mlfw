import { describe, it, expect } from 'vitest';
import { tensor, sum, gelu, silu } from '../../src/index.js';

function gradValues(t) {
  return [...t.grad.data];
}

function geluRef(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

function geluGradRef(x) {
  const c = 0.7978845608028654;
  const ic = 0.044715;
  const inner = c * (x + ic * x * x * x);
  const th = Math.tanh(inner);
  const sech2 = 1 - th * th;
  const dInner = c * (1 + 3 * ic * x * x);
  return 0.5 * (1 + th + x * sech2 * dInner);
}

function sigmoidRef(x) {
  return 1 / (1 + Math.exp(-x));
}

function siluGradRef(x) {
  const s = sigmoidRef(x);
  return s * (1 + x * (1 - s));
}

describe('gelu backward', () => {
  it('matches analytical gradient at x=0', () => {
    const x = tensor([0], { requiresGrad: true });
    gelu(x).backward();
    expect(gradValues(x)[0]).toBeCloseTo(geluGradRef(0), 4);
  });

  it('matches analytical gradient at positive values', () => {
    const x = tensor([1, 2], { requiresGrad: true });
    sum(gelu(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(geluGradRef(1), 3);
    expect(g[1]).toBeCloseTo(geluGradRef(2), 3);
  });

  it('matches analytical gradient at negative values', () => {
    const x = tensor([-1, -2], { requiresGrad: true });
    sum(gelu(x)).backward();
    const g = gradValues(x);
    expect(g[0]).toBeCloseTo(geluGradRef(-1), 3);
    expect(g[1]).toBeCloseTo(geluGradRef(-2), 3);
  });
});

describe('silu backward', () => {
  it('d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))', () => {
    const vals = [0, 1, -1, 2];
    const x = tensor(vals, { requiresGrad: true });
    sum(silu(x)).backward();
    const g = gradValues(x);
    for (let i = 0; i < vals.length; i++) {
      expect(g[i]).toBeCloseTo(siluGradRef(vals[i]), 4);
    }
  });

  it('gradient at x=0 is 0.5', () => {
    const x = tensor([0], { requiresGrad: true });
    silu(x).backward();
    expect(gradValues(x)[0]).toBeCloseTo(0.5, 4);
  });
});
