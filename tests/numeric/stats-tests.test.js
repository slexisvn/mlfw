import { describe, it, expect } from 'vitest';
import { tensor, numeric } from '../../src/index.js';
import { makeRng } from '../../src/ml/_random.js';

function gaussianSample(seed, n, loc = 0, scale = 1) {
  const rng = makeRng(seed);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = rng();
    while (u <= 0) u = rng();
    out[i] = loc + scale * numeric.normalPpfScalar(u);
  }
  return out;
}

function exponentialSample(seed, n) {
  const rng = makeRng(seed);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = rng();
    while (u <= 0) u = rng();
    out[i] = -Math.log(u);
  }
  return out;
}

describe('t-tests', () => {
  it('is zero when the sample mean equals popmean', () => {
    const r = numeric.tTest1Samp([1, 2, 3, 4, 5], { popmean: 3 });
    expect(r.statistic).toBeCloseTo(0, 12);
    expect(r.pvalue).toBeCloseTo(1, 12);
    expect(r.df).toBe(4);
  });

  it('matches the closed-form one-sample statistic and p-value', () => {
    const r = numeric.tTest1Samp([1, 2, 3, 4, 5], { popmean: 2.5 });
    expect(r.statistic).toBeCloseTo(0.7071067811865476, 10);
    expect(r.pvalue).toBeCloseTo(2 * (1 - numeric.studentT.cdf(r.statistic, 4)), 12);
  });

  it('accepts tensor input', () => {
    const x = tensor(new Float64Array([1, 2, 3, 4, 5]), { shape: [5], dtype: 'f64' });
    const r = numeric.tTest1Samp(x, { popmean: 2.5 });
    expect(r.statistic).toBeCloseTo(0.7071067811865476, 10);
  });

  it('matches the pooled two-sample statistic', () => {
    const r = numeric.tTestInd([1, 2, 3, 4], [2, 3, 4, 5]);
    expect(r.statistic).toBeCloseTo(-1.0954451150103321, 10);
    expect(r.df).toBe(6);
  });

  it('Welch equals pooled for equal variances and sizes', () => {
    const pooled = numeric.tTestInd([1, 2, 3, 4], [2, 3, 4, 5]);
    const welch = numeric.tTestInd([1, 2, 3, 4], [2, 3, 4, 5], { equalVar: false });
    expect(welch.statistic).toBeCloseTo(pooled.statistic, 12);
    expect(welch.df).toBeCloseTo(pooled.df, 10);
  });

  it('Welch df drops below pooled df for unequal variances', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [10, 30, -10, 50, -30, 60, -40, 70];
    const welch = numeric.tTestInd(x, y, { equalVar: false });
    const pooled = numeric.tTestInd(x, y);
    expect(welch.df).toBeLessThan(pooled.df);
  });

  it('paired test equals a one-sample test on differences', () => {
    const x = [3.1, 2.8, 4.2, 3.9, 5.0];
    const y = [2.9, 2.9, 3.8, 3.5, 4.6];
    const paired = numeric.tTestPaired(x, y);
    const diffs = x.map((v, i) => v - y[i]);
    const one = numeric.tTest1Samp(diffs);
    expect(paired.statistic).toBeCloseTo(one.statistic, 12);
    expect(paired.pvalue).toBeCloseTo(one.pvalue, 12);
    expect(paired.df).toBe(one.df);
  });
});

describe('chi-square tests', () => {
  it('goodness-of-fit is zero for exactly uniform counts', () => {
    const r = numeric.chi2Gof([10, 10, 10, 10]);
    expect(r.statistic).toBeCloseTo(0, 12);
    expect(r.pvalue).toBeCloseTo(1, 12);
    expect(r.df).toBe(3);
  });

  it('matches the classic uniform goodness-of-fit example', () => {
    const r = numeric.chi2Gof([16, 18, 16, 14, 12, 12]);
    expect(r.statistic).toBeCloseTo(2.0, 10);
    expect(r.df).toBe(5);
    expect(r.pvalue).toBeCloseTo(1 - numeric.chi2.cdf(2, 5), 12);
    expect(r.pvalue).toBeGreaterThan(0.8);
  });

  it('honors explicit expected counts and ddof', () => {
    const r = numeric.chi2Gof([16, 18, 16, 14, 12, 12], [16, 16, 16, 16, 12, 12], { ddof: 1 });
    expect(r.statistic).toBeCloseTo(0.25 + 0.25, 10);
    expect(r.df).toBe(4);
  });

  it('independence is zero for a perfectly independent table', () => {
    const r = numeric.chi2Independence([[10, 20], [30, 60]]);
    expect(r.statistic).toBeCloseTo(0, 10);
    expect(r.pvalue).toBeCloseTo(1, 10);
    expect(r.df).toBe(1);
  });

  it('matches the hand-computed 2x2 dependence statistic', () => {
    const r = numeric.chi2Independence(tensor(new Float64Array([10, 20, 20, 10]), { shape: [2, 2], dtype: 'f64' }));
    expect(r.statistic).toBeCloseTo(100 / 15, 10);
    expect(r.df).toBe(1);
    expect(r.pvalue).toBeCloseTo(1 - numeric.chi2.cdf(100 / 15, 1), 12);
  });
});

describe('Kolmogorov-Smirnov tests', () => {
  it('gives a high p-value for perfect normal scores', () => {
    const n = 100;
    const x = Array.from({ length: n }, (_, i) => numeric.normalPpfScalar((i + 0.5) / n));
    const r = numeric.ksTest1Samp(x);
    expect(r.statistic).toBeCloseTo(0.5 / n, 10);
    expect(r.pvalue).toBeGreaterThan(0.99);
  });

  it('rejects uniform data against a standard normal', () => {
    const n = 200;
    const x = Array.from({ length: n }, (_, i) => (i + 0.5) / n);
    const r = numeric.ksTest1Samp(x);
    expect(r.statistic).toBeGreaterThan(0.4);
    expect(r.pvalue).toBeLessThan(1e-6);
  });

  it('reproduces the asymptotic 5% critical level', () => {
    const n = 100;
    const target = 1.358 / (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n));
    const shift = target - 0.5 / n;
    const x = Array.from({ length: n }, (_, i) => i + 1);
    const r = numeric.ksTest1Samp(x, (v) => (v - 0.5) / n - shift);
    expect(r.statistic).toBeCloseTo(target, 12);
    expect(r.pvalue).toBeCloseTo(0.05, 2);
  });

  it('supports a custom cdf with loc/scale options', () => {
    const x = gaussianSample(7, 500, 3, 2);
    const bad = numeric.ksTest1Samp(x);
    const good = numeric.ksTest1Samp(x, undefined, { loc: 3, scale: 2 });
    expect(bad.pvalue).toBeLessThan(1e-6);
    expect(good.pvalue).toBeGreaterThan(0.05);
  });

  it('two-sample statistic is zero for identical samples', () => {
    const x = [1, 2, 3, 4, 5];
    const r = numeric.ksTest2Samp(x, x);
    expect(r.statistic).toBe(0);
    expect(r.pvalue).toBeCloseTo(1, 10);
  });

  it('two-sample statistic is one for disjoint samples', () => {
    const r = numeric.ksTest2Samp([1, 2, 3, 4, 5], [10, 11, 12, 13, 14]);
    expect(r.statistic).toBeCloseTo(1, 12);
    expect(r.pvalue).toBeLessThan(0.02);
  });

  it('two-sample accepts samples from the same distribution', () => {
    const r = numeric.ksTest2Samp(gaussianSample(11, 400), gaussianSample(13, 350));
    expect(r.pvalue).toBeGreaterThan(0.05);
  });
});

describe('normality tests', () => {
  const gaussian = gaussianSample(24, 3000);
  const skewed = exponentialSample(22, 3000);

  it('Jarque-Bera is small on gaussian data and large on skewed data', () => {
    const g = numeric.jarqueBera(gaussian);
    const s = numeric.jarqueBera(skewed);
    expect(g.pvalue).toBeGreaterThan(0.05);
    expect(s.statistic).toBeGreaterThan(100);
    expect(s.pvalue).toBeLessThan(1e-10);
    expect(g.df).toBe(2);
  });

  it('D\'Agostino K2 agrees on gaussian vs skewed data', () => {
    const g = numeric.dagostinoK2(gaussian);
    const s = numeric.dagostinoK2(skewed);
    expect(g.pvalue).toBeGreaterThan(0.05);
    expect(s.pvalue).toBeLessThan(1e-10);
  });

  it('Anderson-Darling agrees on gaussian vs skewed data', () => {
    const g = numeric.andersonDarling(gaussian);
    const s = numeric.andersonDarling(skewed);
    expect(g.pvalue).toBeGreaterThan(0.05);
    expect(s.statistic).toBeGreaterThan(10);
    expect(s.pvalue).toBeLessThan(0.001);
  });
});

describe('Mann-Whitney U', () => {
  it('matches the hand-computed separated example', () => {
    const r = numeric.mannWhitneyU([1, 2, 3], [4, 5, 6]);
    expect(r.statistic).toBe(0);
    const z = (4.5 - 0.5) / Math.sqrt(5.25);
    expect(r.pvalue).toBeCloseTo(2 * (1 - numeric.normalCdfScalar(z)), 10);
  });

  it('is symmetric around one for identical tied samples', () => {
    const r = numeric.mannWhitneyU([1, 2, 3], [1, 2, 3]);
    expect(r.statistic).toBeCloseTo(4.5, 12);
    expect(r.pvalue).toBeCloseTo(1, 10);
  });

  it('accepts samples from the same distribution', () => {
    const r = numeric.mannWhitneyU(gaussianSample(31, 300), gaussianSample(37, 280));
    expect(r.pvalue).toBeGreaterThan(0.05);
  });

  it('rejects clearly shifted samples', () => {
    const r = numeric.mannWhitneyU(gaussianSample(41, 300), gaussianSample(43, 300, 1));
    expect(r.pvalue).toBeLessThan(1e-6);
  });
});
