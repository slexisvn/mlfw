import { normalCdfScalar } from '../special.js';
import { studentT, chi2 as chi2Dist } from '../distributions.js';
import { hostVector, hostGrid } from '../_array.js';
import type { NumericMatrixInput, NumericVectorInput, ScalarFn, TestResult } from '../types.js';

const KS_SERIES_MAX_TERMS = 100;
const KS_SERIES_EPS = 1e-16;
const KS_LAMBDA_SHIFT = 0.12;
const KS_LAMBDA_SCALE = 0.11;
const AD_CORRECTION = [0.75, 2.25];
const AD_BREAKS = [0.2, 0.34, 0.6];
const AD_P_LOW = [-13.436, 101.14, -223.73];
const AD_P_LOWMID = [-8.318, 42.796, -59.938];
const AD_P_MID = [0.9177, -4.279, -1.38];
const AD_P_HIGH = [1.2937, -5.709, 0.0186];

type Summary = { n: number; mean: number; variance: number };
type Moments = { n: number; mean: number; m2: number; skew: number; kurt: number };
type MeanOptions = { popmean?: number };
type TTestOptions = MeanOptions & { equalVar?: boolean };
type Chi2Options = { ddof?: number };
type KsOptions = { loc?: number; scale?: number };

function summary(v: Float64Array): Summary {
  const n = v.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += v[i];
  const mean = s / n;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = v[i] - mean;
    ss += d * d;
  }
  return { n, mean, variance: ss / (n - 1) };
}

function centralMoments(v: Float64Array): Moments {
  const n = v.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += v[i];
  const mean = s / n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = v[i] - mean;
    const d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  return { n, mean, m2, skew: m3 / Math.pow(m2, 1.5), kurt: m4 / (m2 * m2) };
}

function clampP(p: number): number {
  return Math.min(Math.max(p, 0), 1);
}

function tPvalue(t: number, df: number): number {
  return clampP(2 * (1 - (studentT.cdf(Math.abs(t), df) as number)));
}

function chi2Pvalue(stat: number, df: number): number {
  return clampP(1 - (chi2Dist.cdf(stat, df) as number));
}

export function tTest1Samp(x: NumericVectorInput, opts: MeanOptions = {}): TestResult {
  const { data } = hostVector(x);
  const mu = opts.popmean ?? 0;
  const { n, mean, variance } = summary(data);
  const statistic = (mean - mu) / Math.sqrt(variance / n);
  const df = n - 1;
  return { statistic, pvalue: tPvalue(statistic, df), df };
}

export function tTestInd(x: NumericVectorInput, y: NumericVectorInput, opts: TTestOptions = {}): TestResult {
  const a = summary(hostVector(x).data);
  const b = summary(hostVector(y).data);
  if (opts.equalVar ?? true) {
    const df = a.n + b.n - 2;
    const pooled = ((a.n - 1) * a.variance + (b.n - 1) * b.variance) / df;
    const statistic = (a.mean - b.mean) / Math.sqrt(pooled * (1 / a.n + 1 / b.n));
    return { statistic, pvalue: tPvalue(statistic, df), df };
  }
  const va = a.variance / a.n;
  const vb = b.variance / b.n;
  const statistic = (a.mean - b.mean) / Math.sqrt(va + vb);
  const df = ((va + vb) * (va + vb)) / ((va * va) / (a.n - 1) + (vb * vb) / (b.n - 1));
  return { statistic, pvalue: tPvalue(statistic, df), df };
}

export function tTestPaired(x: NumericVectorInput, y: NumericVectorInput, opts: MeanOptions = {}): TestResult {
  const a = hostVector(x).data;
  const b = hostVector(y).data;
  if (a.length !== b.length) {
    throw new Error(`tTestPaired: samples must have equal length, got ${a.length} and ${b.length}`);
  }
  const d = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) d[i] = a[i] - b[i];
  return tTest1Samp(d, { popmean: opts.popmean ?? 0 });
}

export function chi2Gof(observed: NumericVectorInput, expected?: NumericVectorInput | null, opts: Chi2Options = {}): TestResult {
  const obs = hostVector(observed).data;
  const k = obs.length;
  let exp: Float64Array;
  if (expected === undefined || expected === null) {
    let total = 0;
    for (let i = 0; i < k; i++) total += obs[i];
    exp = new Float64Array(k).fill(total / k);
  } else {
    exp = hostVector(expected).data;
  }
  let statistic = 0;
  for (let i = 0; i < k; i++) {
    const d = obs[i] - exp[i];
    statistic += (d * d) / exp[i];
  }
  const df = k - 1 - (opts.ddof ?? 0);
  return { statistic, pvalue: chi2Pvalue(statistic, df), df };
}

export function chi2Independence(table: NumericMatrixInput): TestResult {
  const { data, rows, cols } = hostGrid(table);
  const rowSums = new Float64Array(rows);
  const colSums = new Float64Array(cols);
  let total = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = data[i * cols + j];
      rowSums[i] += v;
      colSums[j] += v;
      total += v;
    }
  }
  let statistic = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const e = (rowSums[i] * colSums[j]) / total;
      const d = data[i * cols + j] - e;
      statistic += (d * d) / e;
    }
  }
  const df = (rows - 1) * (cols - 1);
  return { statistic, pvalue: chi2Pvalue(statistic, df), df };
}

function kolmogorovSf(lambda: number): number {
  if (lambda <= 0) return 1;
  let sum = 0;
  for (let j = 1; j <= KS_SERIES_MAX_TERMS; j++) {
    const term = Math.exp(-2 * j * j * lambda * lambda);
    sum += j % 2 === 1 ? term : -term;
    if (term < KS_SERIES_EPS) break;
  }
  return clampP(2 * sum);
}

function ksPvalue(d: number, ne: number): number {
  const sq = Math.sqrt(ne);
  return kolmogorovSf(d * (sq + KS_LAMBDA_SHIFT + KS_LAMBDA_SCALE / sq));
}

export function ksTest1Samp(x: NumericVectorInput, cdf?: ScalarFn, opts: KsOptions = {}): Omit<TestResult, 'df'> {
  const { data } = hostVector(x);
  const sorted = Float64Array.from(data).sort();
  const n = sorted.length;
  const loc = opts.loc ?? 0;
  const scale = opts.scale ?? 1;
  const F = typeof cdf === 'function' ? cdf : (v: number) => normalCdfScalar((v - loc) / scale);
  let statistic = 0;
  for (let i = 0; i < n; i++) {
    const f = F(sorted[i]);
    statistic = Math.max(statistic, (i + 1) / n - f, f - i / n);
  }
  return { statistic, pvalue: ksPvalue(statistic, n) };
}

export function ksTest2Samp(x: NumericVectorInput, y: NumericVectorInput): Omit<TestResult, 'df'> {
  const a = Float64Array.from(hostVector(x).data).sort();
  const b = Float64Array.from(hostVector(y).data).sort();
  const n1 = a.length;
  const n2 = b.length;
  let i = 0;
  let j = 0;
  let statistic = 0;
  while (i < n1 && j < n2) {
    const v = Math.min(a[i], b[j]);
    while (i < n1 && a[i] <= v) i++;
    while (j < n2 && b[j] <= v) j++;
    statistic = Math.max(statistic, Math.abs(i / n1 - j / n2));
  }
  const ne = (n1 * n2) / (n1 + n2);
  return { statistic, pvalue: ksPvalue(statistic, ne) };
}

export function jarqueBera(x: NumericVectorInput): TestResult {
  const { n, skew, kurt } = centralMoments(hostVector(x).data);
  const statistic = (n / 6) * (skew * skew + ((kurt - 3) * (kurt - 3)) / 4);
  return { statistic, pvalue: chi2Pvalue(statistic, 2), df: 2 };
}

function skewZ(v: Float64Array): number {
  const { n, skew } = centralMoments(v);
  const y = skew * Math.sqrt(((n + 1) * (n + 3)) / (6 * (n - 2)));
  const beta2 = (3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)) / ((n - 2) * (n + 5) * (n + 7) * (n + 9));
  const w2 = -1 + Math.sqrt(2 * (beta2 - 1));
  const delta = 1 / Math.sqrt(0.5 * Math.log(w2));
  const alpha = Math.sqrt(2 / (w2 - 1));
  const t = y / alpha;
  return delta * Math.log(t + Math.sqrt(t * t + 1));
}

function kurtosisZ(v: Float64Array): number {
  const { n, kurt } = centralMoments(v);
  const eb2 = (3 * (n - 1)) / (n + 1);
  const vb2 = (24 * n * (n - 2) * (n - 3)) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
  const xx = (kurt - eb2) / Math.sqrt(vb2);
  const sqrtBeta1 = ((6 * (n * n - 5 * n + 2)) / ((n + 7) * (n + 9)))
    * Math.sqrt((6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)));
  const a = 6 + (8 / sqrtBeta1) * (2 / sqrtBeta1 + Math.sqrt(1 + 4 / (sqrtBeta1 * sqrtBeta1)));
  const num = 1 - 2 / (9 * a);
  const denom = 1 + xx * Math.sqrt(2 / (a - 4));
  return (num - Math.cbrt((1 - 2 / a) / denom)) / Math.sqrt(2 / (9 * a));
}

export function dagostinoK2(x: NumericVectorInput): TestResult {
  const { data } = hostVector(x);
  const zs = skewZ(data);
  const zk = kurtosisZ(data);
  const statistic = zs * zs + zk * zk;
  return { statistic, pvalue: chi2Pvalue(statistic, 2), df: 2 };
}

function poly2(c: readonly number[], x: number): number {
  return c[0] + c[1] * x + c[2] * x * x;
}

function adPvalue(star: number): number {
  if (star >= AD_BREAKS[2]) return clampP(Math.exp(poly2(AD_P_HIGH, star)));
  if (star > AD_BREAKS[1]) return clampP(Math.exp(poly2(AD_P_MID, star)));
  if (star > AD_BREAKS[0]) return clampP(1 - Math.exp(poly2(AD_P_LOWMID, star)));
  return clampP(1 - Math.exp(poly2(AD_P_LOW, star)));
}

export function andersonDarling(x: NumericVectorInput): Omit<TestResult, 'df'> {
  const { data } = hostVector(x);
  const sorted = Float64Array.from(data).sort();
  const n = sorted.length;
  const { mean, variance } = summary(sorted);
  const sd = Math.sqrt(variance);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const lo = normalCdfScalar((sorted[i] - mean) / sd);
    const hi = normalCdfScalar(-(sorted[n - 1 - i] - mean) / sd);
    s += (2 * i + 1) * (Math.log(lo) + Math.log(hi));
  }
  const statistic = -n - s / n;
  const star = statistic * (1 + AD_CORRECTION[0] / n + AD_CORRECTION[1] / (n * n));
  return { statistic, pvalue: adPvalue(star) };
}

function rankData(v: Float64Array): { ranks: Float64Array; tieSum: number } {
  const n = v.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((p, q) => v[p] - v[q]);
  const ranks = new Float64Array(n);
  let tieSum = 0;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && v[order[j + 1]] === v[order[i]]) j++;
    const rank = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) ranks[order[k]] = rank;
    const t = j - i + 1;
    tieSum += t * t * t - t;
    i = j + 1;
  }
  return { ranks, tieSum };
}

export function mannWhitneyU(x: NumericVectorInput, y: NumericVectorInput): Omit<TestResult, 'df'> {
  const a = hostVector(x).data;
  const b = hostVector(y).data;
  const n1 = a.length;
  const n2 = b.length;
  const combined = new Float64Array(n1 + n2);
  combined.set(a);
  combined.set(b, n1);
  const { ranks, tieSum } = rankData(combined);
  let r1 = 0;
  for (let i = 0; i < n1; i++) r1 += ranks[i];
  const statistic = r1 - (n1 * (n1 + 1)) / 2;
  const n = n1 + n2;
  const mu = (n1 * n2) / 2;
  const sigma = Math.sqrt(((n1 * n2) / 12) * (n + 1 - tieSum / (n * (n - 1))));
  const diff = statistic - mu;
  const z = (diff - 0.5 * Math.sign(diff)) / sigma;
  return { statistic, pvalue: clampP(2 * (1 - normalCdfScalar(Math.abs(z)))) };
}
