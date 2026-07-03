import { fft, ifft } from './transforms.js';
import { lstsq } from '../tensor/ops/linalg.js';
import { hostVector, nextPow2 } from './_array.js';
import { toHostTensor } from '../tensor/utils/host_matrix.js';
import { tensorToContiguous } from '../dispatcher/jit_dispatch.js';

const DK_MAX_ITER = 200;
const DK_TOL = 1e-13;
const DK_SEED_RE = 0.4;
const DK_SEED_IM = 0.9;

function fftInterleaved(data, m, device) {
  const padded = new Float64Array(m);
  padded.set(data);
  return tensorToContiguous(fft(toHostTensor(padded, [m], 'f64', device)));
}

function modeSlice(la, lb, mode) {
  const lFull = la + lb - 1;
  if (mode === 'full') return { offset: 0, length: lFull };
  const lMax = Math.max(la, lb);
  const lMin = Math.min(la, lb);
  if (mode === 'same') return { offset: (lMin - 1) >> 1, length: lMax };
  if (mode === 'valid') return { offset: lMin - 1, length: lMax - lMin + 1 };
  throw new Error(`convolve: unknown mode '${mode}', expected 'full', 'same', or 'valid'`);
}

function fftConvolve(a, b, mode, device, dtype) {
  const la = a.length;
  const lb = b.length;
  const m = nextPow2(la + lb - 1);
  const A = fftInterleaved(a, m, device);
  const B = fftInterleaved(b, m, device);
  const prod = new Float64Array(2 * m);
  for (let k = 0; k < m; k++) {
    const ar = A[2 * k];
    const ai = A[2 * k + 1];
    const br = B[2 * k];
    const bi = B[2 * k + 1];
    prod[2 * k] = ar * br - ai * bi;
    prod[2 * k + 1] = ar * bi + ai * br;
  }
  const C = tensorToContiguous(ifft(toHostTensor(prod, [m, 2], 'f64', device)));
  const { offset, length } = modeSlice(la, lb, mode);
  const out = new Float64Array(length);
  for (let i = 0; i < length; i++) out[i] = C[2 * (offset + i)];
  return toHostTensor(out, [length], dtype, device);
}

export function convolve(a, b, opts = {}) {
  const av = hostVector(a);
  const bv = hostVector(b);
  return fftConvolve(av.data, bv.data, opts.mode ?? 'full', av.device, av.dtype);
}

export function correlate(a, b, opts = {}) {
  const av = hostVector(a);
  const bv = hostVector(b);
  const reversed = new Float64Array(bv.data.length);
  for (let i = 0; i < bv.data.length; i++) reversed[i] = bv.data[bv.data.length - 1 - i];
  return fftConvolve(av.data, reversed, opts.mode ?? 'full', av.device, av.dtype);
}

function checkWindow(window, n) {
  const w = Math.floor(window);
  if (!(w >= 1 && w <= n)) {
    throw new Error(`rolling: window must be in [1, ${n}], got ${window}`);
  }
  return w;
}

export function rollingSum(x, window) {
  const { data, device, dtype } = hostVector(x);
  const n = data.length;
  const w = checkWindow(window, n);
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + data[i];
  const out = new Float64Array(n - w + 1);
  for (let i = 0; i < out.length; i++) out[i] = prefix[i + w] - prefix[i];
  return toHostTensor(out, [out.length], dtype, device);
}

export function rollingMean(x, window) {
  const { data, device, dtype } = hostVector(x);
  const n = data.length;
  const w = checkWindow(window, n);
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + data[i];
  const out = new Float64Array(n - w + 1);
  for (let i = 0; i < out.length; i++) out[i] = (prefix[i + w] - prefix[i]) / w;
  return toHostTensor(out, [out.length], dtype, device);
}

export function rollingStd(x, window, opts = {}) {
  const { data, device, dtype } = hostVector(x);
  const n = data.length;
  const w = checkWindow(window, n);
  const ddof = opts.ddof ?? 1;
  let total = 0;
  for (let i = 0; i < n; i++) total += data[i];
  const mu = total / n;
  const s1 = new Float64Array(n + 1);
  const s2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const d = data[i] - mu;
    s1[i + 1] = s1[i] + d;
    s2[i + 1] = s2[i] + d * d;
  }
  const out = new Float64Array(n - w + 1);
  for (let i = 0; i < out.length; i++) {
    const sum = s1[i + w] - s1[i];
    const sumsq = s2[i + w] - s2[i];
    out[i] = Math.sqrt(Math.max((sumsq - (sum * sum) / w) / (w - ddof), 0));
  }
  return toHostTensor(out, [out.length], dtype, device);
}

function rollingExtreme(x, window, keep) {
  const { data, device, dtype } = hostVector(x);
  const n = data.length;
  const w = checkWindow(window, n);
  const out = new Float64Array(n - w + 1);
  const deque = new Int32Array(n);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < n; i++) {
    while (tail > head && keep(data[i], data[deque[tail - 1]])) tail--;
    deque[tail++] = i;
    if (deque[head] <= i - w) head++;
    if (i >= w - 1) out[i - w + 1] = data[deque[head]];
  }
  return toHostTensor(out, [out.length], dtype, device);
}

export function rollingMin(x, window) {
  return rollingExtreme(x, window, (a, b) => a <= b);
}

export function rollingMax(x, window) {
  return rollingExtreme(x, window, (a, b) => a >= b);
}

export function polyfit(x, y, deg) {
  const xv = hostVector(x);
  const yv = hostVector(y);
  const n = xv.data.length;
  if (yv.data.length !== n) {
    throw new Error(`polyfit: x and y must have equal length, got ${n} and ${yv.data.length}`);
  }
  const cols = deg + 1;
  const V = new Float64Array(n * cols);
  for (let i = 0; i < n; i++) {
    V[i * cols + deg] = 1;
    for (let j = deg - 1; j >= 0; j--) V[i * cols + j] = V[i * cols + j + 1] * xv.data[i];
  }
  const sol = lstsq(
    toHostTensor(V, [n, cols], 'f64', xv.device),
    toHostTensor(yv.data, [n, 1], 'f64', xv.device),
  );
  return toHostTensor(Float64Array.from(tensorToContiguous(sol)), [cols], 'f64', xv.device);
}

function horner(c, x) {
  let acc = 0;
  for (let i = 0; i < c.length; i++) acc = acc * x + c[i];
  return acc;
}

export function polyval(coeffs, x) {
  const c = hostVector(coeffs).data;
  if (typeof x === 'number') return horner(c, x);
  const xv = hostVector(x);
  const out = new Float64Array(xv.data.length);
  for (let i = 0; i < out.length; i++) out[i] = horner(c, xv.data[i]);
  return toHostTensor(out, [out.length], xv.dtype, xv.device);
}

export function polyroots(coeffs) {
  const { data, device } = hostVector(coeffs);
  let start = 0;
  while (start < data.length - 1 && data[start] === 0) start++;
  const deg = data.length - 1 - start;
  if (deg < 1) return toHostTensor(new Float64Array(0), [0, 2], 'f64', device);
  const monic = new Float64Array(deg + 1);
  for (let i = 0; i <= deg; i++) monic[i] = data[start + i] / data[start];
  const rRe = new Float64Array(deg);
  const rIm = new Float64Array(deg);
  let pr = 1;
  let pi = 0;
  for (let k = 0; k < deg; k++) {
    const nr = pr * DK_SEED_RE - pi * DK_SEED_IM;
    pi = pr * DK_SEED_IM + pi * DK_SEED_RE;
    pr = nr;
    rRe[k] = pr;
    rIm[k] = pi;
  }
  for (let iter = 0; iter < DK_MAX_ITER; iter++) {
    let delta = 0;
    for (let k = 0; k < deg; k++) {
      let vr = 1;
      let vi = 0;
      for (let i = 1; i <= deg; i++) {
        const nr2 = vr * rRe[k] - vi * rIm[k] + monic[i];
        vi = vr * rIm[k] + vi * rRe[k];
        vr = nr2;
      }
      let dr = 1;
      let di = 0;
      for (let j = 0; j < deg; j++) {
        if (j === k) continue;
        const er = rRe[k] - rRe[j];
        const ei = rIm[k] - rIm[j];
        const nr3 = dr * er - di * ei;
        di = dr * ei + di * er;
        dr = nr3;
      }
      const denom = dr * dr + di * di;
      const qr = (vr * dr + vi * di) / denom;
      const qi = (vi * dr - vr * di) / denom;
      rRe[k] -= qr;
      rIm[k] -= qi;
      delta = Math.max(delta, Math.hypot(qr, qi));
    }
    if (delta < DK_TOL) break;
  }
  const out = new Float64Array(2 * deg);
  for (let k = 0; k < deg; k++) {
    out[2 * k] = rRe[k];
    out[2 * k + 1] = rIm[k];
  }
  return toHostTensor(out, [deg, 2], 'f64', device);
}
