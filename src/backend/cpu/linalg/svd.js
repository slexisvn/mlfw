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

export function svdHost(a, rows, cols, opts, gramFn = gram) {
  const tol = opts?.tol ?? DEFAULT_TOL;
  const k = Math.min(rows, cols);
  const U = new Float64Array(rows * k);
  const S = new Float64Array(k);
  const V = new Float64Array(cols * k);

  if (cols <= rows) {
    const { values, vectors } = eighHost(gramFn(a, rows, cols, true), cols, opts);
    for (let c = 0; c < k; c++) {
      const idx = cols - 1 - c;
      const sv = Math.sqrt(Math.max(values[idx], 0));
      S[c] = sv;
      for (let r = 0; r < cols; r++) V[r * k + c] = vectors[r * cols + idx];
      if (sv > tol) {
        for (let i = 0; i < rows; i++) {
          let acc = 0;
          for (let j = 0; j < cols; j++) acc += a[i * cols + j] * V[j * k + c];
          U[i * k + c] = acc / sv;
        }
      }
    }
  } else {
    const { values, vectors } = eighHost(gramFn(a, rows, cols, false), rows, opts);
    for (let c = 0; c < k; c++) {
      const idx = rows - 1 - c;
      const sv = Math.sqrt(Math.max(values[idx], 0));
      S[c] = sv;
      for (let r = 0; r < rows; r++) U[r * k + c] = vectors[r * rows + idx];
      if (sv > tol) {
        for (let j = 0; j < cols; j++) {
          let acc = 0;
          for (let i = 0; i < rows; i++) acc += a[i * cols + j] * U[i * k + c];
          V[j * k + c] = acc / sv;
        }
      }
    }
  }
  return { U, S, V, k };
}
