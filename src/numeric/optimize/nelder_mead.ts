import type { VectorFn } from '../types.js';

const DEFAULT_TOL = 1e-10;
const DEFAULT_INITIAL_STEP = 0.05;
const DEFAULT_ZERO_STEP = 0.00025;
const REFLECT = 1;
const EXPAND = 2;
const CONTRACT = 0.5;
const SHRINK = 0.5;
const ITER_PER_DIM = 200;

type NelderMeadOptions = {
  maxIter?: number;
  tol?: number;
  initialStep?: number;
  zeroStep?: number;
  alpha?: number;
  gamma?: number;
  rho?: number;
  sigma?: number;
};
type MinimizeResult = { point: number[]; value: number; iterations: number; converged: boolean };

export function nelderMead(f: VectorFn, x0: readonly number[], opts: NelderMeadOptions = {}): MinimizeResult {
  const n = x0.length;
  const maxIter = opts.maxIter ?? ITER_PER_DIM * n;
  const tol = opts.tol ?? DEFAULT_TOL;
  const spread = opts.initialStep ?? DEFAULT_INITIAL_STEP;
  const zeroStep = opts.zeroStep ?? DEFAULT_ZERO_STEP;
  const alpha = opts.alpha ?? REFLECT;
  const gamma = opts.gamma ?? EXPAND;
  const rho = opts.rho ?? CONTRACT;
  const sigma = opts.sigma ?? SHRINK;

  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    p[i] = p[i] !== 0 ? p[i] * (1 + spread) : zeroStep;
    simplex.push(p);
  }
  const fvals = simplex.map(f);

  const order = Array.from({ length: n + 1 }, (_, i) => i);
  let iterations = 0;
  for (; iterations < maxIter; iterations++) {
    order.sort((a, b) => fvals[a] - fvals[b]);
    const sortedS = order.map((i) => simplex[i]);
    const sortedF = order.map((i) => fvals[i]);
    for (let i = 0; i <= n; i++) {
      simplex[i] = sortedS[i];
      fvals[i] = sortedF[i];
    }
    if (Math.abs(fvals[n] - fvals[0]) < tol) break;

    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i][j];
    for (let j = 0; j < n; j++) centroid[j] /= n;

    const worst = simplex[n];
    const xr = centroid.map((c, j) => c + alpha * (c - worst[j]));
    const fr = f(xr);
    if (fr < fvals[0]) {
      const xe = centroid.map((c, j) => c + gamma * (xr[j] - c));
      const fe = f(xe);
      if (fe < fr) {
        simplex[n] = xe;
        fvals[n] = fe;
      } else {
        simplex[n] = xr;
        fvals[n] = fr;
      }
    } else if (fr < fvals[n - 1]) {
      simplex[n] = xr;
      fvals[n] = fr;
    } else {
      const xc = centroid.map((c, j) => c + rho * (worst[j] - c));
      const fc = f(xc);
      if (fc < fvals[n]) {
        simplex[n] = xc;
        fvals[n] = fc;
      } else {
        const best = simplex[0];
        for (let i = 1; i <= n; i++) {
          simplex[i] = best.map((b: number, j: number) => b + sigma * (simplex[i][j] - b));
          fvals[i] = f(simplex[i]);
        }
      }
    }
  }

  let best = 0;
  for (let i = 1; i <= n; i++) if (fvals[i] < fvals[best]) best = i;
  return { point: simplex[best], value: fvals[best], iterations, converged: iterations < maxIter };
}
