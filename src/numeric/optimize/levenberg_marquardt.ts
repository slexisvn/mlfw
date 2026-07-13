import { solve } from '../../tensor/ops/linalg.js';
import { toHostTensor } from '../../tensor/utils/host_matrix.js';
import { dot, DEFAULT_FD_STEP } from './_util.js';
import type { JacobianFn, ResidualFn } from '../types.js';

const DEFAULT_MAX_ITER = 200;
const DEFAULT_TOL = 1e-10;
const DEFAULT_LAMBDA = 1e-3;
const LAMBDA_UP = 10;
const LAMBDA_DOWN = 10;
const LAMBDA_MAX = 1e12;

type LMOptions = {
  maxIter?: number;
  tol?: number;
  step?: number;
  jacobian?: JacobianFn;
  lambda?: number;
};
type MinimizeResult = { point: number[]; value: number; iterations: number; converged: boolean };

function numericJacobian(residual: ResidualFn, x: readonly number[], m: number, step: number): number[][] {
  const n = x.length;
  const J = new Array<number[]>(m);
  for (let i = 0; i < m; i++) J[i] = new Array<number>(n);
  const p = x.slice();
  for (let j = 0; j < n; j++) {
    const xj = p[j];
    p[j] = xj + step;
    const rp = residual(p);
    p[j] = xj - step;
    const rm = residual(p);
    p[j] = xj;
    for (let i = 0; i < m; i++) J[i][j] = (rp[i] - rm[i]) / (2 * step);
  }
  return J;
}

function solveNormal(JtJ: number[][], Jtr: readonly number[], lambda: number, n: number): number[] {
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i * n + j] = JtJ[i][j];
    A[i * n + i] += lambda * JtJ[i][i];
  }
  const b = Float64Array.from(Jtr);
  const x = solve(toHostTensor(A, [n, n], 'f64'), toHostTensor(b, [n], 'f64'));
  return Array.from(x.toArray() as ArrayLike<number>);
}

export function levenbergMarquardt(residual: ResidualFn, x0: readonly number[], opts: LMOptions = {}): MinimizeResult {
  const maxIter = opts.maxIter ?? DEFAULT_MAX_ITER;
  const tol = opts.tol ?? DEFAULT_TOL;
  const step = opts.step ?? DEFAULT_FD_STEP;
  const jacobian = opts.jacobian ?? ((x: number[], m: number) => numericJacobian(residual, x, m, step));

  let x = x0.slice();
  let r = residual(x);
  const m = r.length;
  const n = x.length;
  let cost = dot(r, r);
  let lambda = opts.lambda ?? DEFAULT_LAMBDA;

  let iterations = 0;
  for (; iterations < maxIter; iterations++) {
    const J = jacobian(x, m);
    const JtJ = new Array<number[]>(n);
    const Jtr = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      JtJ[i] = new Array<number>(n).fill(0);
      for (let k = 0; k < m; k++) Jtr[i] += J[k][i] * r[k];
    }
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let s = 0;
        for (let k = 0; k < m; k++) s += J[k][i] * J[k][j];
        JtJ[i][j] = s;
        JtJ[j][i] = s;
      }
    }

    let improved = false;
    while (lambda < LAMBDA_MAX) {
      let delta: number[];
      try {
        delta = solveNormal(JtJ, Jtr, lambda, n);
      } catch (_) {
        lambda *= LAMBDA_UP;
        continue;
      }
      const cand = x.map((v, j) => v - delta[j]);
      const rc = residual(cand);
      const cc = dot(rc, rc);
      if (cc < cost) {
        x = cand;
        r = rc;
        const delta_cost = cost - cc;
        cost = cc;
        lambda /= LAMBDA_DOWN;
        improved = true;
        if (delta_cost < tol * (1 + cost)) {
          return { point: x, value: cost, iterations: iterations + 1, converged: true };
        }
        break;
      }
      lambda *= LAMBDA_UP;
    }
    if (!improved) return { point: x, value: cost, iterations, converged: cost < tol };
  }
  return { point: x, value: cost, iterations, converged: false };
}
