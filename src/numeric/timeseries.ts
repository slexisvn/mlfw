import { fft, ifft } from './transforms.js';
import { chi2 as chi2Dist } from './distributions.js';
import { hostVector } from './_array.js';
import { f64, nextPow2 } from '../util/numeric_array.js';
import { toHostTensor } from '../tensor/utils/host_matrix.js';
import { tensorToContiguous } from '../dispatcher/jit_dispatch.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { Device } from '../tensor/types/device.js';
import type { NumericVectorInput, TestResult } from './types.js';

type AcfOptions = { nlags?: number };
type LjungBoxOptions = { lags?: number; modelDf?: number };
type PeriodogramOptions = { detrend?: boolean };

function defaultNlags(n: number): number {
  return Math.max(1, Math.min(n - 1, Math.floor(10 * Math.log10(n))));
}

function autocovariance(data: Float64Array, device: Device, nlags: number): Float64Array {
  const n = data.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += data[i];
  const mean = s / n;
  const m = nextPow2(2 * n);
  const padded = new Float64Array(m);
  for (let i = 0; i < n; i++) padded[i] = data[i] - mean;
  const F = tensorToContiguous(fft(toHostTensor(padded, [m], 'f64', device)));
  const spec = new Float64Array(2 * m);
  for (let k = 0; k < m; k++) {
    spec[2 * k] = f64(F, 2 * k) * f64(F, 2 * k) + f64(F, 2 * k + 1) * f64(F, 2 * k + 1);
  }
  const C = tensorToContiguous(ifft(toHostTensor(spec, [m, 2], 'f64', device)));
  const acov = new Float64Array(nlags + 1);
  for (let k = 0; k <= nlags; k++) acov[k] = f64(C, 2 * k) / n;
  return acov;
}

function normalizedAcf(data: Float64Array, device: Device, nlags: number): Float64Array {
  const acov = autocovariance(data, device, nlags);
  const r = new Float64Array(nlags + 1);
  for (let k = 0; k <= nlags; k++) r[k] = acov[k] / acov[0];
  return r;
}

export function acf(x: NumericVectorInput, opts: AcfOptions = {}): Tensor {
  const { data, device, dtype } = hostVector(x);
  const nlags = opts.nlags ?? defaultNlags(data.length);
  const r = normalizedAcf(data, device, nlags);
  return toHostTensor(r, [nlags + 1], dtype, device);
}

export function pacf(x: NumericVectorInput, opts: AcfOptions = {}): Tensor {
  const { data, device, dtype } = hostVector(x);
  const nlags = opts.nlags ?? defaultNlags(data.length);
  const r = normalizedAcf(data, device, nlags);
  const out = new Float64Array(nlags + 1);
  out[0] = 1;
  let prev = new Float64Array(nlags + 1);
  let cur = new Float64Array(nlags + 1);
  for (let k = 1; k <= nlags; k++) {
    let num = r[k];
    let den = 1;
    for (let j = 1; j < k; j++) {
      num -= prev[j] * r[k - j];
      den -= prev[j] * r[j];
    }
    const a = num / den;
    cur[k] = a;
    for (let j = 1; j < k; j++) cur[j] = prev[j] - a * prev[k - j];
    out[k] = a;
    const tmp = prev;
    prev = cur;
    cur = tmp;
  }
  return toHostTensor(out, [nlags + 1], dtype, device);
}

export function ljungBox(x: NumericVectorInput, opts: LjungBoxOptions = {}): TestResult {
  const { data, device } = hostVector(x);
  const n = data.length;
  const lags = opts.lags ?? Math.min(10, n - 1);
  const r = normalizedAcf(data, device, lags);
  let statistic = 0;
  for (let k = 1; k <= lags; k++) statistic += (r[k] * r[k]) / (n - k);
  statistic *= n * (n + 2);
  const df = lags - (opts.modelDf ?? 0);
  return { statistic, pvalue: Math.min(Math.max(1 - (chi2Dist.cdf(statistic, df) as number), 0), 1), df };
}

export function durbinWatson(x: NumericVectorInput): number {
  const { data } = hostVector(x);
  let num = 0;
  let den = 0;
  for (let i = 0; i < data.length; i++) {
    den += data[i] * data[i];
    if (i > 0) {
      const d = data[i] - data[i - 1];
      num += d * d;
    }
  }
  return num / den;
}

export function periodogram(x: NumericVectorInput, opts: PeriodogramOptions = {}): Tensor {
  const { data, device, dtype } = hostVector(x);
  const n = data.length;
  let signal = data;
  if (opts.detrend !== false) {
    let s = 0;
    for (let i = 0; i < n; i++) s += data[i];
    const mean = s / n;
    signal = new Float64Array(n);
    for (let i = 0; i < n; i++) signal[i] = data[i] - mean;
  }
  const F = tensorToContiguous(fft(toHostTensor(signal, [n], 'f64', device)));
  const m = Math.floor(n / 2);
  const power = new Float64Array(m + 1);
  for (let k = 0; k <= m; k++) {
    power[k] = (f64(F, 2 * k) * f64(F, 2 * k) + f64(F, 2 * k + 1) * f64(F, 2 * k + 1)) / n;
  }
  return toHostTensor(power, [m + 1], dtype, device);
}
