import { describe, it, expect } from 'vitest';
import { tensor, numeric } from '../../src/index.js';

const { normal, studentT, chi2, fisherF } = numeric;

const P_GRID = [0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995];

describe('distributions', () => {
  it('normal cdf/ppf on tensors preserve shape and device', () => {
    const x = tensor(new Float64Array([-1.96, 0, 1.96]), { shape: [3], dtype: 'f64' });
    const c = normal.cdf(x);
    expect(c.shape).toEqual([3]);
    const vals = c.toArray();
    expect(vals[0]).toBeCloseTo(0.024997895148, 6);
    expect(vals[1]).toBeCloseTo(0.5, 8);
    expect(vals[2]).toBeCloseTo(0.975002104852, 6);
  });

  it('normal loc/scale', () => {
    expect(normal.cdf(10, { loc: 10, scale: 3 })).toBeCloseTo(0.5, 8);
    expect(normal.ppf(0.975, { loc: 1, scale: 2 })).toBeCloseTo(1 + 2 * 1.959963984540054, 5);
    expect(normal.pdf(0, { scale: 2 })).toBeCloseTo(1 / (2 * Math.sqrt(2 * Math.PI)), 10);
  });

  it('normal cdf∘ppf identity', () => {
    for (const p of P_GRID) {
      expect(normal.cdf(normal.ppf(p))).toBeCloseTo(p, 8);
    }
  });

  it('t distribution matches tabulated quantiles', () => {
    expect(studentT.ppf(0.975, 20)).toBeCloseTo(2.085963447265837, 5);
    expect(studentT.ppf(0.95, 10)).toBeCloseTo(1.8124611228107335, 5);
    expect(studentT.cdf(0, 7)).toBeCloseTo(0.5, 10);
    for (const p of P_GRID) {
      expect(studentT.cdf(studentT.ppf(p, 12), 12)).toBeCloseTo(p, 7);
    }
  });

  it('chi2 distribution matches tabulated quantiles', () => {
    expect(chi2.ppf(0.95, 10)).toBeCloseTo(18.307038053275146, 5);
    expect(chi2.ppf(0.05, 5)).toBeCloseTo(1.145476226061769, 5);
    for (const p of P_GRID) {
      expect(chi2.cdf(chi2.ppf(p, 4), 4)).toBeCloseTo(p, 7);
    }
  });

  it('F distribution matches tabulated quantiles', () => {
    expect(fisherF.ppf(0.95, 5, 10)).toBeCloseTo(3.325834529905508, 4);
    expect(fisherF.cdf(1, 8, 8)).toBeCloseTo(0.5, 8);
    for (const p of P_GRID) {
      expect(fisherF.cdf(fisherF.ppf(p, 6, 14), 6, 14)).toBeCloseTo(p, 7);
    }
  });

  it('pdf integrates to cdf differences', () => {
    const { quadrature } = numeric;
    const area = quadrature((z) => normal.pdf(z), -1, 1);
    expect(area).toBeCloseTo(normal.cdf(1) - normal.cdf(-1), 6);
    const areaT = quadrature((z) => studentT.pdf(z, 9), 0, 2);
    expect(areaT).toBeCloseTo(studentT.cdf(2, 9) - 0.5, 8);
  });
});
