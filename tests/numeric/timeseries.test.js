import { describe, it, expect } from 'vitest';
import { tensor, numeric } from '../../src/index.js';
import { makeRng } from '../../src/ml/_random.js';

function gaussianNoise(seed, n) {
  const rng = makeRng(seed);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = rng();
    while (u <= 0) u = rng();
    out[i] = numeric.normalPpfScalar(u);
  }
  return out;
}

function arProcess(seed, n, coeffs) {
  const burn = 200;
  const e = gaussianNoise(seed, n + burn);
  const x = new Float64Array(n + burn);
  for (let t = 0; t < n + burn; t++) {
    let v = e[t];
    for (let j = 0; j < coeffs.length; j++) {
      if (t > j) v += coeffs[j] * x[t - 1 - j];
    }
    x[t] = v;
  }
  return x.slice(burn);
}

function directAcf(x, nlags) {
  const n = x.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += x[i];
  const mean = s / n;
  const c = new Float64Array(nlags + 1);
  for (let k = 0; k <= nlags; k++) {
    for (let t = 0; t + k < n; t++) c[k] += (x[t] - mean) * (x[t + k] - mean);
    c[k] /= n;
  }
  return Array.from(c, (v) => v / c[0]);
}

describe('acf', () => {
  it('matches a direct O(n^2) reference exactly', () => {
    const x = arProcess(3, 64, [0.6]);
    const got = numeric.acf(x, { nlags: 12 }).toArray();
    const ref = directAcf(x, 12);
    for (let k = 0; k <= 12; k++) expect(got[k]).toBeCloseTo(ref[k], 10);
  });

  it('decays at the AR(1) coefficient', () => {
    const phi = 0.8;
    const x = arProcess(5, 20000, [phi]);
    const r = numeric.acf(x, { nlags: 5 }).toArray();
    expect(r[0]).toBeCloseTo(1, 12);
    for (let k = 1; k <= 5; k++) expect(r[k]).toBeCloseTo(Math.pow(phi, k), 1);
  });

  it('returns a tensor on the input device with default lag count', () => {
    const x = tensor(gaussianNoise(9, 100), { shape: [100], dtype: 'f64' });
    const r = numeric.acf(x);
    expect(r.shape).toEqual([21]);
    expect(r.device.type).toBe(x.device.type);
  });
});

describe('pacf', () => {
  it('cuts off after lag 1 for an AR(1) process', () => {
    const x = arProcess(7, 20000, [0.8]);
    const p = numeric.pacf(x, { nlags: 5 }).toArray();
    expect(p[0]).toBe(1);
    expect(p[1]).toBeCloseTo(0.8, 1);
    for (let k = 2; k <= 5; k++) expect(Math.abs(p[k])).toBeLessThan(0.05);
  });

  it('cuts off after lag 2 for an AR(2) process', () => {
    const x = arProcess(11, 40000, [0.5, 0.3]);
    const p = numeric.pacf(x, { nlags: 6 }).toArray();
    expect(p[2]).toBeCloseTo(0.3, 1);
    for (let k = 3; k <= 6; k++) expect(Math.abs(p[k])).toBeLessThan(0.05);
  });
});

describe('ljungBox', () => {
  it('keeps a high p-value on white noise', () => {
    const r = numeric.ljungBox(gaussianNoise(13, 2000), { lags: 10 });
    expect(r.df).toBe(10);
    expect(r.pvalue).toBeGreaterThan(0.05);
  });

  it('rejects an autocorrelated series', () => {
    const r = numeric.ljungBox(arProcess(15, 2000, [0.8]), { lags: 10 });
    expect(r.pvalue).toBeLessThan(1e-10);
    expect(r.statistic).toBeGreaterThan(100);
  });

  it('subtracts fitted model degrees of freedom', () => {
    const r = numeric.ljungBox(gaussianNoise(17, 500), { lags: 10, modelDf: 2 });
    expect(r.df).toBe(8);
  });
});

describe('durbinWatson', () => {
  it('matches a hand-computed value', () => {
    expect(numeric.durbinWatson([1, 2, 3, 4])).toBeCloseTo(0.1, 12);
  });

  it('is near two for white noise and small for positive autocorrelation', () => {
    expect(numeric.durbinWatson(gaussianNoise(19, 5000))).toBeCloseTo(2, 1);
    expect(numeric.durbinWatson(arProcess(23, 5000, [0.8]))).toBeLessThan(1);
  });
});

describe('periodogram', () => {
  it('peaks at a planted frequency', () => {
    const n = 256;
    const k0 = 20;
    const x = Array.from({ length: n }, (_, t) => Math.sin((2 * Math.PI * k0 * t) / n) + 0.5);
    const p = numeric.periodogram(x).toArray();
    expect(p.length).toBe(n / 2 + 1);
    let best = 0;
    for (let k = 1; k < p.length; k++) {
      if (p[k] > p[best]) best = k;
    }
    expect(best).toBe(k0);
    expect(p[0]).toBeCloseTo(0, 8);
  });

  it('keeps the mean component when detrend is disabled', () => {
    const x = Array.from({ length: 64 }, () => 1);
    const p = numeric.periodogram(x, { detrend: false }).toArray();
    expect(p[0]).toBeCloseTo(64, 8);
  });
});
