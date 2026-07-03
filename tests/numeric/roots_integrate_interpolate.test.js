import { describe, it, expect } from 'vitest';
import { numeric } from '../../src/index.js';

const { bisect, newton, brentq, trapezoid, simpson, quadrature, linearInterp, cubicSpline } = numeric;

describe('root finding', () => {
  it('bisect finds cos root', () => {
    const r = bisect(Math.cos, 1, 2);
    expect(r.converged).toBe(true);
    expect(r.root).toBeCloseTo(Math.PI / 2, 10);
  });

  it('newton with analytic derivative', () => {
    const r = newton((x) => x * x - 2, 1, { derivative: (x) => 2 * x });
    expect(r.root).toBeCloseTo(Math.SQRT2, 10);
  });

  it('newton with finite-difference fallback', () => {
    const r = newton((x) => Math.exp(x) - 3, 0);
    expect(r.root).toBeCloseTo(Math.log(3), 9);
  });

  it('brentq handles hard roots', () => {
    const f = (x) => x ** 3 - 2 * x - 5;
    const r = brentq(f, 2, 3);
    expect(r.converged).toBe(true);
    expect(f(r.root)).toBeCloseTo(0, 9);
    expect(r.root).toBeCloseTo(2.0945514815423265, 9);
  });

  it('bracket validation throws on same-sign endpoints', () => {
    expect(() => bisect((x) => x * x + 1, -1, 1)).toThrow();
    expect(() => brentq((x) => x * x + 1, -1, 1)).toThrow();
  });
});

describe('quadrature', () => {
  it('trapezoid and simpson converge on smooth integrals', () => {
    expect(trapezoid(Math.sin, 0, Math.PI, { n: 4096 })).toBeCloseTo(2, 6);
    expect(simpson(Math.sin, 0, Math.PI)).toBeCloseTo(2, 10);
    expect(simpson((x) => x * x, 0, 1)).toBeCloseTo(1 / 3, 12);
  });

  it('adaptive quadrature reaches tight tolerance', () => {
    expect(quadrature(Math.sin, 0, Math.PI)).toBeCloseTo(2, 10);
    expect(quadrature((x) => 4 / (1 + x * x), 0, 1)).toBeCloseTo(Math.PI, 10);
    expect(quadrature((x) => Math.exp(-x * x), -6, 6)).toBeCloseTo(Math.sqrt(Math.PI), 9);
  });
});

describe('interpolation', () => {
  it('linearInterp interpolates and clamps', () => {
    const xs = [0, 1, 2];
    const ys = [0, 10, 20];
    expect(linearInterp(xs, ys, 0.5)).toBeCloseTo(5, 12);
    expect(linearInterp(xs, ys, 1.25)).toBeCloseTo(12.5, 12);
    expect(linearInterp(xs, ys, -1)).toBe(0);
    expect(linearInterp(xs, ys, 5)).toBe(20);
    expect(linearInterp(xs, ys, [0.5, 1.5])).toEqual([5, 15]);
  });

  it('cubicSpline passes through knots', () => {
    const xs = [0, 1, 2, 3, 4];
    const ys = xs.map((x) => Math.sin(x));
    const sp = cubicSpline(xs, ys);
    for (let i = 0; i < xs.length; i++) {
      expect(sp.evaluate(xs[i])).toBeCloseTo(ys[i], 12);
    }
  });

  it('cubicSpline is C1-smooth at interior knots', () => {
    const xs = [0, 1, 2, 3, 4];
    const ys = [0, 2, 1, 3, 0];
    const sp = cubicSpline(xs, ys);
    const h = 1e-6;
    for (const k of [1, 2, 3]) {
      const left = (sp.evaluate(k) - sp.evaluate(k - h)) / h;
      const right = (sp.evaluate(k + h) - sp.evaluate(k)) / h;
      expect(left).toBeCloseTo(right, 4);
    }
  });

  it('natural boundary second derivative vanishes at ends', () => {
    const xs = [0, 1, 2, 3];
    const ys = [1, 4, 2, 5];
    const sp = cubicSpline(xs, ys);
    expect(sp.coefficients[0]).toBeCloseTo(0, 12);
    expect(sp.coefficients[3]).toBeCloseTo(0, 12);
  });

  it('cubicSpline approximates a smooth function between knots', () => {
    const xs = Array.from({ length: 21 }, (_, i) => i / 5);
    const ys = xs.map((x) => Math.cos(x));
    const sp = cubicSpline(xs, ys);
    for (const xq of [1.57, 2.13, 2.9]) {
      expect(sp.evaluate(xq)).toBeCloseTo(Math.cos(xq), 5);
    }
    for (const xq of [0.13, 3.77]) {
      expect(sp.evaluate(xq)).toBeCloseTo(Math.cos(xq), 2);
    }
  });
});
