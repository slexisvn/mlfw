import { svdHost } from './svd.js';
import { DEFAULT_RCOND } from './config.js';
import type { LinalgOpts } from './config.js';

export function lstsqHost(a: Float64Array, rows: number, cols: number, b: Float64Array, nrhs: number, opts?: LinalgOpts): Float64Array {
  const rcond = opts?.rcond ?? DEFAULT_RCOND;
  const { U, S, V, k } = svdHost(a, rows, cols, opts);
  const cutoff = rcond * (S.length ? S[0] : 0);

  const t = new Float64Array(k * nrhs);
  for (let c = 0; c < k; c++) {
    const sv = S[c];
    for (let r = 0; r < nrhs; r++) {
      let s = 0;
      for (let i = 0; i < rows; i++) s += U[i * k + c] * b[i * nrhs + r];
      t[c * nrhs + r] = sv > cutoff ? s / sv : 0;
    }
  }

  const x = new Float64Array(cols * nrhs);
  for (let j = 0; j < cols; j++) {
    for (let r = 0; r < nrhs; r++) {
      let s = 0;
      for (let c = 0; c < k; c++) s += V[j * k + c] * t[c * nrhs + r];
      x[j * nrhs + r] = s;
    }
  }
  return x;
}

export function pinvHost(a: Float64Array, rows: number, cols: number, opts?: LinalgOpts): Float64Array {
  const rcond = opts?.rcond ?? DEFAULT_RCOND;
  const { U, S, V, k } = svdHost(a, rows, cols, opts);
  const cutoff = rcond * (S.length ? S[0] : 0);
  const out = new Float64Array(cols * rows);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      let s = 0;
      for (let c = 0; c < k; c++) {
        const sv = S[c];
        if (sv > cutoff) s += V[j * k + c] * (1 / sv) * U[i * k + c];
      }
      out[j * rows + i] = s;
    }
  }
  return out;
}
