import { Tensor } from '../tensor/core/tensor.js';
import { tensorToContiguous } from '../dispatcher/jit_dispatch.js';
import { toHostTensor } from '../tensor/utils/host_matrix.js';
import { brentq } from './roots.js';
import {
  normalCdfScalar, normalPdfScalar, normalPpfScalar,
  lowerGammaRegularized, betaRegularized, lgammaScalar,
} from './special.js';

const PPF_REFINE_STEPS = 3;
const PPF_BRACKET_GROWTH = 2;
const PPF_BRACKET_MAX = 60;
const PPF_TOL = 1e-11;

function mapElementwise(x, fn) {
  if (x instanceof Tensor) {
    const data = tensorToContiguous(x);
    const out = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = fn(data[i]);
    return toHostTensor(out, x.shape, x.dtype, x.device);
  }
  return fn(x);
}

function invertCdf(cdf, pdf, p, seed, opts = {}) {
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
  cdf: (x, opts = {}) => {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return mapElementwise(x, (v) => normalCdfScalar((v - loc) / scale));
  },
  pdf: (x, opts = {}) => {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return mapElementwise(x, (v) => normalPdfScalar((v - loc) / scale) / scale);
  },
  ppf: (p, opts = {}) => {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return mapElementwise(p, (v) => loc + scale * normalPpfScalar(v, opts));
  },
};

function tCdfScalar(x, df) {
  if (x === 0) return 0.5;
  const ib = betaRegularized(df / 2, 0.5, df / (df + x * x));
  return x > 0 ? 1 - ib / 2 : ib / 2;
}

function tPdfScalar(x, df) {
  const logc = lgammaScalar((df + 1) / 2) - lgammaScalar(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(logc - ((df + 1) / 2) * Math.log(1 + (x * x) / df));
}

export const studentT = {
  cdf: (x, df, opts = {}) => mapElementwise(x, (v) => tCdfScalar(v, df ?? opts.df)),
  pdf: (x, df, opts = {}) => mapElementwise(x, (v) => tPdfScalar(v, df ?? opts.df)),
  ppf: (p, df, opts = {}) => {
    const k = df ?? opts.df;
    return mapElementwise(p, (v) => invertCdf(
      (z) => tCdfScalar(z, k),
      (z) => tPdfScalar(z, k),
      v,
      normalPpfScalar(v),
      opts,
    ));
  },
};

function chi2CdfScalar(x, df) {
  if (x <= 0) return 0;
  return lowerGammaRegularized(df / 2, x / 2);
}

function chi2PdfScalar(x, df) {
  if (x <= 0) return 0;
  const half = df / 2;
  return Math.exp((half - 1) * Math.log(x) - x / 2 - half * Math.log(2) - lgammaScalar(half));
}

function chi2Seed(p, df) {
  const z = normalPpfScalar(p);
  const c = 2 / (9 * df);
  const w = 1 - c + z * Math.sqrt(c);
  return Math.max(df * w * w * w, Number.EPSILON);
}

export const chi2 = {
  cdf: (x, df, opts = {}) => mapElementwise(x, (v) => chi2CdfScalar(v, df ?? opts.df)),
  pdf: (x, df, opts = {}) => mapElementwise(x, (v) => chi2PdfScalar(v, df ?? opts.df)),
  ppf: (p, df, opts = {}) => {
    const k = df ?? opts.df;
    return mapElementwise(p, (v) => invertCdf(
      (z) => chi2CdfScalar(z, k),
      (z) => chi2PdfScalar(z, k),
      v,
      chi2Seed(v, k),
      { ...opts, lowerLimit: 0 },
    ));
  },
};

function fCdfScalar(x, d1, d2) {
  if (x <= 0) return 0;
  return betaRegularized(d1 / 2, d2 / 2, (d1 * x) / (d1 * x + d2));
}

function fPdfScalar(x, d1, d2) {
  if (x <= 0) return 0;
  const logB = lgammaScalar(d1 / 2) + lgammaScalar(d2 / 2) - lgammaScalar((d1 + d2) / 2);
  return Math.exp(
    (d1 / 2) * Math.log(d1 / d2) + (d1 / 2 - 1) * Math.log(x)
    - ((d1 + d2) / 2) * Math.log(1 + (d1 / d2) * x) - logB,
  );
}

export const fisherF = {
  cdf: (x, d1, d2, opts = {}) => mapElementwise(x, (v) => fCdfScalar(v, d1 ?? opts.d1, d2 ?? opts.d2)),
  pdf: (x, d1, d2, opts = {}) => mapElementwise(x, (v) => fPdfScalar(v, d1 ?? opts.d1, d2 ?? opts.d2)),
  ppf: (p, d1, d2, opts = {}) => {
    const a = d1 ?? opts.d1;
    const b = d2 ?? opts.d2;
    return mapElementwise(p, (v) => invertCdf(
      (z) => fCdfScalar(z, a, b),
      (z) => fPdfScalar(z, a, b),
      v,
      1,
      { ...opts, lowerLimit: 0 },
    ));
  },
};
