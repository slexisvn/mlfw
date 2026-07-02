import { eighHost } from './jacobi_eigh.js';
import { DEFAULT_TOL } from './config.js';

export function gram(a, rows, cols, transposeLeft) {
  if (transposeLeft) {
    const g = new Float64Array(cols * cols);
    for (let i = 0; i < cols; i++) {
      for (let j = i; j < cols; j++) {
        let s = 0;
        for (let t = 0; t < rows; t++) s += a[t * cols + i] * a[t * cols + j];
        g[i * cols + j] = s;
        g[j * cols + i] = s;
      }
    }
    return g;
  }
  const g = new Float64Array(rows * rows);
  for (let i = 0; i < rows; i++) {
    for (let j = i; j < rows; j++) {
      let s = 0;
      for (let t = 0; t < cols; t++) s += a[i * cols + t] * a[j * cols + t];
      g[i * rows + j] = s;
      g[j * rows + i] = s;
    }
  }
  return g;
}

export function matmulRows(a, m, len, b, p) {
  const out = new Float64Array(m * p);
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < p; c++) {
      let s = 0;
      for (let t = 0; t < len; t++) s += a[i * len + t] * b[c * len + t];
      out[i * p + c] = s;
    }
  }
  return out;
}

export function svdHost(a, rows, cols, opts, gramFn = gram, matmulFn = matmulRows) {
  const tol = opts?.tol ?? DEFAULT_TOL;
  const k = Math.min(rows, cols);
  const U = new Float64Array(rows * k);
  const S = new Float64Array(k);
  const V = new Float64Array(cols * k);

  if (cols <= rows) {
    const { values, vectors } = eighHost(gramFn(a, rows, cols, true), cols, opts);
    const vt = new Float64Array(k * cols);
    for (let c = 0; c < k; c++) {
      const idx = cols - 1 - c;
      S[c] = Math.sqrt(Math.max(values[idx], 0));
      for (let r = 0; r < cols; r++) {
        const v = vectors[r * cols + idx];
        V[r * k + c] = v;
        vt[c * cols + r] = v;
      }
    }
    const proj = matmulFn(a, rows, cols, vt, k);
    for (let c = 0; c < k; c++) {
      if (S[c] > tol) {
        for (let i = 0; i < rows; i++) U[i * k + c] = proj[i * k + c] / S[c];
      }
    }
  } else {
    const { values, vectors } = eighHost(gramFn(a, rows, cols, false), rows, opts);
    const ut = new Float64Array(k * rows);
    for (let c = 0; c < k; c++) {
      const idx = rows - 1 - c;
      S[c] = Math.sqrt(Math.max(values[idx], 0));
      for (let r = 0; r < rows; r++) {
        const u = vectors[r * rows + idx];
        U[r * k + c] = u;
        ut[c * rows + r] = u;
      }
    }
    const at = new Float64Array(cols * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) at[j * rows + i] = a[i * cols + j];
    }
    const proj = matmulFn(at, cols, rows, ut, k);
    for (let c = 0; c < k; c++) {
      if (S[c] > tol) {
        for (let j = 0; j < cols; j++) V[j * k + c] = proj[j * k + c] / S[c];
      }
    }
  }
  return { U, S, V, k };
}
