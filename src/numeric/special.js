import {
  erfScalar, erfcScalar, lgammaScalar, gammaScalar, digammaScalar,
} from '../util/special_math.js';

export { erfScalar, erfcScalar, lgammaScalar, gammaScalar, digammaScalar };

const PPF_A = [
  -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
  1.383577518672690e2, -3.066479806614716e1, 2.506628277459239,
];
const PPF_B = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
  6.680131188771972e1, -1.328068155288572e1,
];
const PPF_C = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
  -2.549732539343734, 4.374664141464968, 2.938163982698783,
];
const PPF_D = [
  7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416,
];
const PPF_LOW = 0.02425;

const CF_MAX_ITER = 300;
const CF_EPS = 3e-14;
const CF_TINY = 1e-300;
const HALLEY_STEPS = 2;

export const SQRT2 = Math.SQRT2;
export const SQRT_2PI = Math.sqrt(2 * Math.PI);
export const INV_SQRT_PI = 1 / Math.sqrt(Math.PI);

export function lowerGammaRegularized(a, x) {
  if (x <= 0) return 0;
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let n = 1; n < CF_MAX_ITER; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * CF_EPS) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - lgammaScalar(a));
  }
  let b = x + 1 - a;
  let c = 1 / CF_TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < CF_MAX_ITER; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < CF_TINY) d = CF_TINY;
    c = b + an / c;
    if (Math.abs(c) < CF_TINY) c = CF_TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < CF_EPS) break;
  }
  return 1 - h * Math.exp(-x + a * Math.log(x) - lgammaScalar(a));
}

function betacf(a, b, x) {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < CF_TINY) d = CF_TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m < CF_MAX_ITER; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < CF_TINY) d = CF_TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < CF_TINY) c = CF_TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < CF_TINY) d = CF_TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < CF_TINY) c = CF_TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < CF_EPS) break;
  }
  return h;
}

export function betaRegularized(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(
    lgammaScalar(a + b) - lgammaScalar(a) - lgammaScalar(b) + a * Math.log(x) + b * Math.log(1 - x),
  );
  if (x < (a + 1) / (a + b + 2)) return (front * betacf(a, b, x)) / a;
  return 1 - (front * betacf(b, a, 1 - x)) / b;
}

export function normalCdfScalar(z) {
  return 0.5 * erfcScalar(-z / SQRT2);
}

export function normalPdfScalar(z) {
  return Math.exp(-0.5 * z * z) / SQRT_2PI;
}

function ppfSeed(p) {
  if (p < PPF_LOW) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((PPF_C[0] * q + PPF_C[1]) * q + PPF_C[2]) * q + PPF_C[3]) * q + PPF_C[4]) * q + PPF_C[5])
      / ((((PPF_D[0] * q + PPF_D[1]) * q + PPF_D[2]) * q + PPF_D[3]) * q + 1);
  }
  if (p > 1 - PPF_LOW) {
    return -ppfSeed(1 - p);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((PPF_A[0] * r + PPF_A[1]) * r + PPF_A[2]) * r + PPF_A[3]) * r + PPF_A[4]) * r + PPF_A[5]) * q
    / (((((PPF_B[0] * r + PPF_B[1]) * r + PPF_B[2]) * r + PPF_B[3]) * r + PPF_B[4]) * r + 1);
}

export function normalPpfScalar(p, opts = {}) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const steps = opts.refineSteps ?? HALLEY_STEPS;
  let z = ppfSeed(p);
  for (let i = 0; i < steps; i++) {
    const e = normalCdfScalar(z) - p;
    const u = e / normalPdfScalar(z);
    z -= u / (1 + (z * u) / 2);
  }
  return z;
}
