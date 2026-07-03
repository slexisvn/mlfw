import { describe, it, expect } from 'vitest';
import { tensor, numeric } from '../../src/index.js';
import { makeRng } from '../../src/ml/_random.js';

function randomArray(seed, n) {
  const rng = makeRng(seed);
  return Array.from({ length: n }, () => rng() * 2 - 1);
}

function directConvolve(a, b) {
  const out = new Float64Array(a.length + b.length - 1);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  }
  return out;
}

function sliceMode(full, la, lb, mode) {
  if (mode === 'full') return Array.from(full);
  const lMax = Math.max(la, lb);
  const lMin = Math.min(la, lb);
  if (mode === 'same') return Array.from(full.slice((lMin - 1) >> 1, ((lMin - 1) >> 1) + lMax));
  return Array.from(full.slice(lMin - 1, lMin - 1 + (lMax - lMin + 1)));
}

describe('convolve / correlate', () => {
  it('matches the classic small example', () => {
    const got = numeric.convolve([1, 2, 3], [0, 1, 0.5]).toArray();
    const want = [0, 1, 2.5, 4, 1.5];
    for (let i = 0; i < want.length; i++) expect(got[i]).toBeCloseTo(want[i], 10);
  });

  it('matches a direct reference for all modes and orderings', () => {
    const a = randomArray(3, 25);
    const b = randomArray(5, 9);
    for (const [x, y] of [[a, b], [b, a]]) {
      const full = directConvolve(x, y);
      for (const mode of ['full', 'same', 'valid']) {
        const got = numeric.convolve(x, y, { mode }).toArray();
        const want = sliceMode(full, x.length, y.length, mode);
        expect(got.length).toBe(want.length);
        for (let i = 0; i < want.length; i++) expect(got[i]).toBeCloseTo(want[i], 9);
      }
    }
  });

  it('correlate matches convolution with the reversed kernel', () => {
    const a = randomArray(7, 20);
    const b = randomArray(11, 6);
    const full = directConvolve(a, Array.from(b).reverse());
    for (const mode of ['full', 'same', 'valid']) {
      const got = numeric.correlate(a, b, { mode }).toArray();
      const want = sliceMode(full, a.length, b.length, mode);
      for (let i = 0; i < want.length; i++) expect(got[i]).toBeCloseTo(want[i], 9);
    }
  });

  it('returns tensors on the input device', () => {
    const a = tensor(new Float64Array([1, 2, 3]), { shape: [3], dtype: 'f64' });
    const out = numeric.convolve(a, [1, 1]);
    expect(out.shape).toEqual([4]);
    expect(out.device.type).toBe(a.device.type);
  });

  it('rejects unknown modes', () => {
    expect(() => numeric.convolve([1, 2], [1], { mode: 'bogus' })).toThrow(/mode/);
  });
});

describe('rolling reductions', () => {
  const x = randomArray(13, 200).map((v, i) => (i % 7 === 0 ? 0.25 : v));
  const w = 12;

  function brute(fn) {
    const out = [];
    for (let i = 0; i + w <= x.length; i++) out.push(fn(x.slice(i, i + w)));
    return out;
  }

  it('rollingSum and rollingMean match brute force', () => {
    const sums = numeric.rollingSum(x, w).toArray();
    const means = numeric.rollingMean(x, w).toArray();
    const refSums = brute((s) => s.reduce((p, v) => p + v, 0));
    expect(sums.length).toBe(x.length - w + 1);
    for (let i = 0; i < refSums.length; i++) {
      expect(sums[i]).toBeCloseTo(refSums[i], 9);
      expect(means[i]).toBeCloseTo(refSums[i] / w, 9);
    }
  });

  it('rollingStd matches brute force with both ddofs', () => {
    for (const ddof of [0, 1]) {
      const got = numeric.rollingStd(x, w, { ddof }).toArray();
      const ref = brute((s) => {
        const m = s.reduce((p, v) => p + v, 0) / s.length;
        return Math.sqrt(s.reduce((p, v) => p + (v - m) * (v - m), 0) / (s.length - ddof));
      });
      for (let i = 0; i < ref.length; i++) expect(got[i]).toBeCloseTo(ref[i], 9);
    }
  });

  it('rollingStd matches a hand-computed window', () => {
    const got = numeric.rollingStd([1, 2, 3, 4], 2).toArray();
    for (const v of got) expect(v).toBeCloseTo(Math.SQRT1_2, 10);
  });

  it('rollingMin and rollingMax match brute force with duplicates', () => {
    const mins = numeric.rollingMin(x, w).toArray();
    const maxs = numeric.rollingMax(x, w).toArray();
    const refMins = brute((s) => Math.min(...s));
    const refMaxs = brute((s) => Math.max(...s));
    for (let i = 0; i < refMins.length; i++) {
      expect(mins[i]).toBe(refMins[i]);
      expect(maxs[i]).toBe(refMaxs[i]);
    }
  });

  it('rejects invalid windows', () => {
    expect(() => numeric.rollingMean([1, 2, 3], 0)).toThrow(/window/);
    expect(() => numeric.rollingMean([1, 2, 3], 4)).toThrow(/window/);
  });
});

describe('polynomials', () => {
  it('polyfit recovers a planted polynomial exactly', () => {
    const coeffs = [2, -3, 1];
    const xs = Array.from({ length: 9 }, (_, i) => i - 4);
    const ys = xs.map((v) => 2 * v * v - 3 * v + 1);
    const got = numeric.polyfit(xs, ys, 2).toArray();
    for (let i = 0; i < coeffs.length; i++) expect(got[i]).toBeCloseTo(coeffs[i], 8);
  });

  it('polyfit smooths noisy samples close to the truth', () => {
    const rng = makeRng(17);
    const xs = Array.from({ length: 200 }, (_, i) => (i - 100) / 25);
    const ys = xs.map((v) => 0.5 * v * v * v - v + 2 + (rng() - 0.5) * 0.01);
    const got = numeric.polyfit(xs, ys, 3).toArray();
    const want = [0.5, 0, -1, 2];
    for (let i = 0; i < want.length; i++) expect(got[i]).toBeCloseTo(want[i], 2);
  });

  it('polyval evaluates by Horner for scalars, arrays, and tensors', () => {
    const c = [1, -6, 11, -6];
    expect(numeric.polyval(c, 0)).toBeCloseTo(-6, 12);
    const arr = numeric.polyval(c, [1, 2, 3]).toArray();
    for (const v of arr) expect(v).toBeCloseTo(0, 10);
    const t = tensor(new Float64Array([4]), { shape: [1], dtype: 'f64' });
    expect(numeric.polyval(c, t).toArray()[0]).toBeCloseTo(6, 10);
  });

  it('polyroots finds real roots of a cubic', () => {
    const roots = numeric.polyroots([1, -6, 11, -6]).toArray();
    const reals = roots.map((r) => r[0]).sort((p, q) => p - q);
    expect(reals[0]).toBeCloseTo(1, 8);
    expect(reals[1]).toBeCloseTo(2, 8);
    expect(reals[2]).toBeCloseTo(3, 8);
    for (const r of roots) expect(Math.abs(r[1])).toBeLessThan(1e-8);
  });

  it('polyroots finds complex roots of x^2 + 1', () => {
    const roots = numeric.polyroots([1, 0, 1]).toArray();
    const imags = roots.map((r) => r[1]).sort((p, q) => p - q);
    expect(imags[0]).toBeCloseTo(-1, 8);
    expect(imags[1]).toBeCloseTo(1, 8);
    for (const r of roots) expect(Math.abs(r[0])).toBeLessThan(1e-8);
  });

  it('polyroots trims leading zero coefficients', () => {
    const roots = numeric.polyroots([0, 1, -1]).toArray();
    expect(roots.length).toBe(1);
    expect(roots[0][0]).toBeCloseTo(1, 10);
  });
});
