import { Tensor } from '../tensor/core/tensor.js';
import { tensorToContiguous } from '../dispatcher/jit_dispatch.js';
import { toHostTensor } from '../tensor/utils/host_matrix.js';
import { brentq } from './roots.js';
import {
  normalCdfScalar, normalPdfScalar, normalPpfScalar,
  lowerGammaRegularized, betaRegularized, lgammaScalar,
} from './special.js';
import type { NumericElementInput, ScalarFn } from './types.js';

const PPF_REFINE_STEPS = 3;
const PPF_BRACKET_GROWTH = 2;
const PPF_BRACKET_MAX = 60;
const PPF_TOL = 1e-11;

type DistOptions = { loc?: number; scale?: number; df?: number; d1?: number; d2?: number; refineSteps?: number; tol?: number; lowerLimit?: number };

function mapElementwise(x: NumericElementInput, fn: ScalarFn): number | Tensor {
  if (x instanceof Tensor) {
    const data = tensorToContiguous(x);
    const out = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = fn(Number(data[i]));
    return toHostTensor(out, x.shape, x.dtype, x.device);
  }
  return fn(x);
}

function invertCdf(cdf: ScalarFn, pdf: ScalarFn, p: number, seed: number, opts: DistOptions = {}): number {
  if (p <= 0 || p >= 1) {
    if (p === 0) return opts.lowerLimit ?? -Infinity;
    if (p === 1) return Infinity;
    return NaN;
  }
  const steps = opts.refineSteps ?? PPF_REFINE_STEPS;
  const tol = opts.tol ?? PPF_TOL;
  let x = seed;
  for (let i = 0; i < steps; i++) {
    const e = cdf(x) - p;
    const d = pdf(x);
    if (!Number.isFinite(x) || d <= 0) break;
    const xn = x - e / d;
    x = opts.lowerLimit !== undefined && xn <= opts.lowerLimit ? (x + opts.lowerLimit) / 2 : xn;
    if (Math.abs(e) < tol) return x;
  }
  if (Math.abs(cdf(x) - p) < tol) return x;
  let lo = opts.lowerLimit !== undefined ? opts.lowerLimit + Number.EPSILON : x;
  let hi = x;
  let width = Math.max(1, Math.abs(x));
  for (let i = 0; i < PPF_BRACKET_MAX; i++) {
    if (opts.lowerLimit === undefined) lo = x - width;
    hi = x + width;
    if (cdf(lo) - p < 0 && cdf(hi) - p > 0) {
      return brentq((z) => cdf(z) - p, lo, hi, { tol }).root;
    }
    width *= PPF_BRACKET_GROWTH;
  }
  return x;
}

export const normal = {
  cdf: (x: NumericElementInput, opts: DistOptions = {}) => {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return mapElementwise(x, (v: number) => normalCdfScalar((v - loc) / scale));
  },
  pdf: (x: NumericElementInput, opts: DistOptions = {}) => {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return mapElementwise(x, (v: number) => normalPdfScalar((v - loc) / scale) / scale);
  },
  ppf: (p: NumericElementInput, opts: DistOptions = {}) => {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return mapElementwise(p, (v: number) => loc + scale * normalPpfScalar(v, opts));
  },
};

function tCdfScalar(x: number, df: number): number {
  if (x === 0) return 0.5;
  const ib = betaRegularized(df / 2, 0.5, df / (df + x * x));
  return x > 0 ? 1 - ib / 2 : ib / 2;
}

function tPdfScalar(x: number, df: number): number {
  const logc = lgammaScalar((df + 1) / 2) - lgammaScalar(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(logc - ((df + 1) / 2) * Math.log(1 + (x * x) / df));
}

export const studentT = {
  cdf: (x: NumericElementInput, df?: number, opts: DistOptions = {}) => mapElementwise(x, (v: number) => tCdfScalar(v, (df ?? opts.df) as number)),
  pdf: (x: NumericElementInput, df?: number, opts: DistOptions = {}) => mapElementwise(x, (v: number) => tPdfScalar(v, (df ?? opts.df) as number)),
  ppf: (p: NumericElementInput, df?: number, opts: DistOptions = {}) => {
    const k = (df ?? opts.df) as number;
    return mapElementwise(p, (v: number) => invertCdf(
      (z: number) => tCdfScalar(z, k),
      (z: number) => tPdfScalar(z, k),
      v,
      normalPpfScalar(v),
      opts,
    ));
  },
};

function chi2CdfScalar(x: number, df: number): number {
  if (x <= 0) return 0;
  return lowerGammaRegularized(df / 2, x / 2);
}

function chi2PdfScalar(x: number, df: number): number {
  if (x <= 0) return 0;
  const half = df / 2;
  return Math.exp((half - 1) * Math.log(x) - x / 2 - half * Math.log(2) - lgammaScalar(half));
}

function chi2Seed(p: number, df: number): number {
  const z = normalPpfScalar(p);
  const c = 2 / (9 * df);
  const w = 1 - c + z * Math.sqrt(c);
  return Math.max(df * w * w * w, Number.EPSILON);
}

export const chi2 = {
  cdf: (x: NumericElementInput, df?: number, opts: DistOptions = {}) => mapElementwise(x, (v: number) => chi2CdfScalar(v, (df ?? opts.df) as number)),
  pdf: (x: NumericElementInput, df?: number, opts: DistOptions = {}) => mapElementwise(x, (v: number) => chi2PdfScalar(v, (df ?? opts.df) as number)),
  ppf: (p: NumericElementInput, df?: number, opts: DistOptions = {}) => {
    const k = (df ?? opts.df) as number;
    return mapElementwise(p, (v: number) => invertCdf(
      (z: number) => chi2CdfScalar(z, k),
      (z: number) => chi2PdfScalar(z, k),
      v,
      chi2Seed(v, k),
      { ...opts, lowerLimit: 0 },
    ));
  },
};

function fCdfScalar(x: number, d1: number, d2: number): number {
  if (x <= 0) return 0;
  return betaRegularized(d1 / 2, d2 / 2, (d1 * x) / (d1 * x + d2));
}

function fPdfScalar(x: number, d1: number, d2: number): number {
  if (x <= 0) return 0;
  const logB = lgammaScalar(d1 / 2) + lgammaScalar(d2 / 2) - lgammaScalar((d1 + d2) / 2);
  return Math.exp(
    (d1 / 2) * Math.log(d1 / d2) + (d1 / 2 - 1) * Math.log(x)
    - ((d1 + d2) / 2) * Math.log(1 + (d1 / d2) * x) - logB,
  );
}

export const fisherF = {
  cdf: (x: NumericElementInput, d1?: number, d2?: number, opts: DistOptions = {}) => mapElementwise(x, (v: number) => fCdfScalar(v, (d1 ?? opts.d1) as number, (d2 ?? opts.d2) as number)),
  pdf: (x: NumericElementInput, d1?: number, d2?: number, opts: DistOptions = {}) => mapElementwise(x, (v: number) => fPdfScalar(v, (d1 ?? opts.d1) as number, (d2 ?? opts.d2) as number)),
  ppf: (p: NumericElementInput, d1?: number, d2?: number, opts: DistOptions = {}) => {
    const a = (d1 ?? opts.d1) as number;
    const b = (d2 ?? opts.d2) as number;
    return mapElementwise(p, (v: number) => invertCdf(
      (z: number) => fCdfScalar(z, a, b),
      (z: number) => fPdfScalar(z, a, b),
      v,
      1,
      { ...opts, lowerLimit: 0 },
    ));
  },
};
