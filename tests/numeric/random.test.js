import { describe, it, expect } from 'vitest';
import { numeric } from '../../src/index.js';

function stats(arr) {
  const n = arr.length;
  let s = 0;
  for (const v of arr) s += v;
  const mean = s / n;
  let ss = 0;
  for (const v of arr) ss += (v - mean) * (v - mean);
  return { mean, variance: ss / (n - 1) };
}

const N = 20000;

describe('Generator reproducibility', () => {
  it('identical seeds produce identical draws', () => {
    const a = new numeric.Generator(42).normal([100]).toArray();
    const b = new numeric.Generator(42).normal([100]).toArray();
    expect(a).toEqual(b);
  });

  it('different seeds produce different draws', () => {
    const a = new numeric.Generator(1).uniform([100]).toArray();
    const b = new numeric.Generator(2).uniform([100]).toArray();
    expect(a).not.toEqual(b);
  });

  it('supports scalar and multi-dimensional shapes', () => {
    expect(new numeric.Generator(3).uniform(5).shape).toEqual([5]);
    expect(new numeric.Generator(3).normal([3, 4]).shape).toEqual([3, 4]);
  });
});

describe('distribution moments', () => {
  it('uniform stays in range with matching moments', () => {
    const draws = new numeric.Generator(7).uniform([N], { low: 2, high: 5 }).toArray();
    for (const v of draws) {
      expect(v).toBeGreaterThanOrEqual(2);
      expect(v).toBeLessThan(5);
    }
    const { mean, variance } = stats(draws);
    expect(mean).toBeCloseTo(3.5, 1);
    expect(variance).toBeCloseTo(0.75, 1);
  });

  it('normal matches loc and scale', () => {
    const draws = new numeric.Generator(11).normal([N], { loc: -1, scale: 2 }).toArray();
    const { mean, variance } = stats(draws);
    expect(mean).toBeCloseTo(-1, 1);
    expect(Math.sqrt(variance)).toBeCloseTo(2, 1);
  });

  it('standard t has variance df / (df - 2)', () => {
    const draws = new numeric.Generator(13).standardT([N], { df: 10 }).toArray();
    const { mean, variance } = stats(draws);
    expect(mean).toBeCloseTo(0, 1);
    expect(variance).toBeCloseTo(1.25, 0.5);
  });

  it('chi2 has mean df and variance 2 df', () => {
    const draws = new numeric.Generator(17).chi2([N], { df: 5 }).toArray();
    const { mean, variance } = stats(draws);
    expect(mean).toBeCloseTo(5, 1);
    expect(Math.abs(variance - 10)).toBeLessThan(0.6);
  });

  it('chi2 sampling handles df below two', () => {
    const draws = new numeric.Generator(19).chi2([N], { df: 1 }).toArray();
    const { mean } = stats(draws);
    for (const v of draws) expect(v).toBeGreaterThan(0);
    expect(mean).toBeCloseTo(1, 1);
  });

  it('exponential has mean and std equal to scale', () => {
    const draws = new numeric.Generator(23).exponential([N], { scale: 2 }).toArray();
    const { mean, variance } = stats(draws);
    expect(mean).toBeCloseTo(2, 1);
    expect(Math.abs(variance - 4)).toBeLessThan(0.3);
  });
});

describe('multivariateNormal', () => {
  const mean = [1, -2];
  const cov = [[2, 0.6], [0.6, 1]];

  it('reproduces the requested mean and covariance', () => {
    const draws = new numeric.Generator(29).multivariateNormal(mean, cov, 30000);
    expect(draws.shape).toEqual([30000, 2]);
    const rows = draws.toArray();
    const m = [0, 0];
    for (const r of rows) {
      m[0] += r[0];
      m[1] += r[1];
    }
    m[0] /= rows.length;
    m[1] /= rows.length;
    expect(m[0]).toBeCloseTo(1, 1);
    expect(m[1]).toBeCloseTo(-2, 1);
    const c = [[0, 0], [0, 0]];
    for (const r of rows) {
      const d0 = r[0] - m[0];
      const d1 = r[1] - m[1];
      c[0][0] += d0 * d0;
      c[0][1] += d0 * d1;
      c[1][1] += d1 * d1;
    }
    expect(c[0][0] / (rows.length - 1)).toBeCloseTo(2, 1);
    expect(c[0][1] / (rows.length - 1)).toBeCloseTo(0.6, 1);
    expect(c[1][1] / (rows.length - 1)).toBeCloseTo(1, 1);
  });

  it('is reproducible for a fixed seed', () => {
    const a = new numeric.Generator(31).multivariateNormal(mean, cov, 4).toArray();
    const b = new numeric.Generator(31).multivariateNormal(mean, cov, 4).toArray();
    expect(a).toEqual(b);
  });

  it('rejects mismatched mean and covariance sizes', () => {
    expect(() => new numeric.Generator(1).multivariateNormal([0], cov, 2)).toThrow(/incompatible/);
  });
});
