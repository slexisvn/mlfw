import { describe, it, expect } from 'vitest';
import { numeric } from '../../src/index.js';

const {
  erfScalar, erfcScalar, lgammaScalar, gammaScalar, digammaScalar,
  normalCdfScalar, normalPpfScalar, lowerGammaRegularized, betaRegularized,
} = numeric;

describe('special functions vs tabulated references', () => {
  it('erf matches known values', () => {
    expect(erfScalar(0)).toBeCloseTo(0, 6);
    expect(erfScalar(0.5)).toBeCloseTo(0.5204998778130465, 6);
    expect(erfScalar(1)).toBeCloseTo(0.8427007929497149, 6);
    expect(erfScalar(2)).toBeCloseTo(0.9953222650189527, 6);
    expect(erfScalar(-1)).toBeCloseTo(-0.8427007929497149, 6);
  });

  it('erfc complements erf', () => {
    for (const x of [-2, -0.5, 0, 0.7, 1.8]) {
      expect(erfcScalar(x) + erfScalar(x)).toBeCloseTo(1, 12);
    }
  });

  it('lgamma matches known values', () => {
    expect(lgammaScalar(1)).toBeCloseTo(0, 10);
    expect(lgammaScalar(2)).toBeCloseTo(0, 10);
    expect(lgammaScalar(0.5)).toBeCloseTo(Math.log(Math.sqrt(Math.PI)), 10);
    expect(lgammaScalar(5)).toBeCloseTo(Math.log(24), 10);
    expect(lgammaScalar(10.5)).toBeCloseTo(13.940625219403763, 8);
  });

  it('gamma satisfies recurrence and reflection', () => {
    expect(gammaScalar(5)).toBeCloseTo(24, 8);
    expect(gammaScalar(0.5)).toBeCloseTo(Math.sqrt(Math.PI), 10);
    for (const x of [0.3, 1.7, 3.2]) {
      expect(gammaScalar(x + 1)).toBeCloseTo(x * gammaScalar(x), 8);
    }
    expect(gammaScalar(-0.5)).toBeCloseTo(-2 * Math.sqrt(Math.PI), 8);
  });

  it('digamma matches known values', () => {
    const euler = 0.5772156649015329;
    expect(digammaScalar(1)).toBeCloseTo(-euler, 10);
    expect(digammaScalar(0.5)).toBeCloseTo(-euler - 2 * Math.log(2), 10);
    expect(digammaScalar(2)).toBeCloseTo(1 - euler, 10);
  });

  it('regularized incomplete gamma known values', () => {
    expect(lowerGammaRegularized(1, 1)).toBeCloseTo(1 - Math.exp(-1), 10);
    expect(lowerGammaRegularized(0.5, 0.5)).toBeCloseTo(erfScalar(Math.sqrt(0.5)), 6);
  });

  it('regularized incomplete beta symmetry and known values', () => {
    expect(betaRegularized(2, 2, 0.5)).toBeCloseTo(0.5, 10);
    expect(betaRegularized(1, 1, 0.3)).toBeCloseTo(0.3, 10);
    for (const [a, b, x] of [[2, 5, 0.3], [0.5, 0.5, 0.7]]) {
      expect(betaRegularized(a, b, x) + betaRegularized(b, a, 1 - x)).toBeCloseTo(1, 10);
    }
  });

  it('normal cdf and ppf match tabulated quantiles', () => {
    expect(normalCdfScalar(0)).toBeCloseTo(0.5, 8);
    expect(normalCdfScalar(1.96)).toBeCloseTo(0.9750021048517795, 6);
    expect(normalPpfScalar(0.975)).toBeCloseTo(1.959963984540054, 5);
    expect(normalPpfScalar(0.995)).toBeCloseTo(2.5758293035489004, 5);
    expect(normalPpfScalar(0.5)).toBeCloseTo(0, 8);
    expect(normalPpfScalar(0.001)).toBeCloseTo(-3.090232306167813, 4);
  });
});
