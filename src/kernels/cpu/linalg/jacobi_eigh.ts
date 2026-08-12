import { MAX_JACOBI_SWEEPS, DEFAULT_TOL, JACOBI_ZERO } from './config.js';
import type { LinalgOpts } from './config.js';

export type EighResult = { values: Float64Array; vectors: Float64Array };

export function eighHost(a: Float64Array, n: number, opts?: LinalgOpts): EighResult {
  const tol = opts?.tol ?? DEFAULT_TOL;
  const maxSweeps = opts?.maxSweeps ?? MAX_JACOBI_SWEEPS;
  const A = Float64Array.from(a);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    }
    if (Math.sqrt(off) < tol) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < JACOBI_ZERO) continue;
        const beta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(beta || 1) / (Math.abs(beta) + Math.sqrt(beta * beta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p];
          const akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k];
          const aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p];
          const vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[i * n + i] - A[j * n + j]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = order[col];
    values[col] = A[src * n + src];
    for (let row = 0; row < n; row++) vectors[row * n + col] = V[row * n + src];
  }
  return { values, vectors };
}
