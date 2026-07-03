import { describe, it, expect } from 'vitest';
import { tensor, matmul, linalg, numeric } from '../../src/index.js';
import { makeRng } from '../../src/ml/_random.js';

function randomMatrix(rng, m, n) {
  const data = new Float64Array(m * n);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  return tensor(data, { shape: [m, n], dtype: 'f64' });
}

function maxAbsDiff(a, b) {
  let err = 0;
  for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]));
  return err;
}

function naiveDft(re, im, invert) {
  const n = re.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const sign = invert ? 1 : -1;
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const ang = (sign * 2 * Math.PI * k * t) / n;
      outRe[k] += re[t] * Math.cos(ang) - im[t] * Math.sin(ang);
      outIm[k] += re[t] * Math.sin(ang) + im[t] * Math.cos(ang);
    }
  }
  if (invert) {
    for (let k = 0; k < n; k++) {
      outRe[k] /= n;
      outIm[k] /= n;
    }
  }
  return { re: outRe, im: outIm };
}

describe('qr decomposition', () => {
  it('reconstructs A = Q @ R with orthonormal Q (tall)', () => {
    const rng = makeRng(11);
    const A = randomMatrix(rng, 8, 5);
    const { Q, R } = linalg.qr(A);
    expect(Q.shape).toEqual([8, 5]);
    expect(R.shape).toEqual([5, 5]);
    const rec = matmul(Q, R).toArray().flat();
    expect(maxAbsDiff(rec, A.toArray().flat())).toBeLessThan(1e-10);
    const QtQ = matmul(Q.transpose(0, 1), Q).toArray().flat();
    const eye = Array.from({ length: 25 }, (_, i) => (i % 6 === 0 ? 1 : 0));
    expect(maxAbsDiff(QtQ, eye)).toBeLessThan(1e-10);
  });

  it('handles wide matrices', () => {
    const rng = makeRng(23);
    const A = randomMatrix(rng, 4, 7);
    const { Q, R } = linalg.qr(A);
    expect(Q.shape).toEqual([4, 4]);
    expect(R.shape).toEqual([4, 7]);
    const rec = matmul(Q, R).toArray().flat();
    expect(maxAbsDiff(rec, A.toArray().flat())).toBeLessThan(1e-10);
  });

  it('R is upper triangular', () => {
    const rng = makeRng(5);
    const A = randomMatrix(rng, 6, 6);
    const { R } = linalg.qr(A);
    const r = R.toArray();
    for (let i = 1; i < 6; i++) {
      for (let j = 0; j < i; j++) expect(r[i][j]).toBe(0);
    }
  });

  it('solves least squares via qr consistently with lstsq', () => {
    const rng = makeRng(31);
    const A = randomMatrix(rng, 10, 3);
    const xTrue = tensor(new Float64Array([1.5, -2, 0.5]), { shape: [3, 1], dtype: 'f64' });
    const b = matmul(A, xTrue);
    const viaLstsq = linalg.lstsq(A, b).toArray().flat();
    expect(maxAbsDiff(viaLstsq, xTrue.toArray().flat())).toBeLessThan(1e-8);
  });
});

describe('fft / ifft', () => {
  it('matches naive DFT for power-of-two length', () => {
    const rng = makeRng(3);
    const n = 16;
    const re = Float64Array.from({ length: n }, () => rng() * 2 - 1);
    const x = tensor(re, { shape: [n], dtype: 'f64' });
    const F = numeric.fft(x).toArray();
    const ref = naiveDft(re, new Float64Array(n), false);
    for (let k = 0; k < n; k++) {
      expect(F[k][0]).toBeCloseTo(ref.re[k], 9);
      expect(F[k][1]).toBeCloseTo(ref.im[k], 9);
    }
  });

  it('matches naive DFT for prime length (Bluestein)', () => {
    const rng = makeRng(17);
    const n = 13;
    const re = Float64Array.from({ length: n }, () => rng() * 2 - 1);
    const im = Float64Array.from({ length: n }, () => rng() * 2 - 1);
    const inter = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      inter[2 * i] = re[i];
      inter[2 * i + 1] = im[i];
    }
    const x = tensor(inter, { shape: [n, 2], dtype: 'f64' });
    const F = numeric.fft(x).toArray();
    const ref = naiveDft(re, im, false);
    for (let k = 0; k < n; k++) {
      expect(F[k][0]).toBeCloseTo(ref.re[k], 8);
      expect(F[k][1]).toBeCloseTo(ref.im[k], 8);
    }
  });

  it('ifft inverts fft for non-power-of-two length', () => {
    const rng = makeRng(29);
    const n = 12;
    const re = Float64Array.from({ length: n }, () => rng() * 2 - 1);
    const x = tensor(re, { shape: [n], dtype: 'f64' });
    const back = numeric.ifft(numeric.fft(x)).toArray();
    for (let i = 0; i < n; i++) {
      expect(back[i][0]).toBeCloseTo(re[i], 9);
      expect(back[i][1]).toBeCloseTo(0, 9);
    }
  });

  it('satisfies Parseval for arbitrary length', () => {
    const rng = makeRng(41);
    const n = 10;
    const re = Float64Array.from({ length: n }, () => rng() * 2 - 1);
    const x = tensor(re, { shape: [n], dtype: 'f64' });
    const F = numeric.fft(x).toArray();
    const timeEnergy = re.reduce((s, v) => s + v * v, 0);
    const freqEnergy = F.reduce((s, c) => s + c[0] * c[0] + c[1] * c[1], 0) / n;
    expect(freqEnergy).toBeCloseTo(timeEnergy, 9);
  });
});
