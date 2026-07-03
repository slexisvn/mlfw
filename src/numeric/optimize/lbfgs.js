import { dot, norm, sub, scale, numericGradient, clampToBounds, DEFAULT_FD_STEP } from './_util.js';

const DEFAULT_MEMORY = 10;
const DEFAULT_MAX_ITER = 1000;
const DEFAULT_GTOL = 1e-8;
const DEFAULT_FTOL = 1e-12;
const WOLFE_C1 = 1e-4;
const BACKTRACK_MIN = 0.1;
const BACKTRACK_MAX = 0.5;
const MAX_LINE_SEARCH = 40;

function twoLoop(g, sList, yList, rhoList) {
  const q = g.slice();
  const m = sList.length;
  const alphas = new Array(m);
  for (let i = m - 1; i >= 0; i--) {
    alphas[i] = rhoList[i] * dot(sList[i], q);
    for (let j = 0; j < q.length; j++) q[j] -= alphas[i] * yList[i][j];
  }
  if (m > 0) {
    const s = sList[m - 1];
    const y = yList[m - 1];
    const gammaScale = dot(s, y) / dot(y, y);
    for (let j = 0; j < q.length; j++) q[j] *= gammaScale;
  }
  for (let i = 0; i < m; i++) {
    const beta = rhoList[i] * dot(yList[i], q);
    for (let j = 0; j < q.length; j++) q[j] += (alphas[i] - beta) * sList[i][j];
  }
  return q;
}

function minimize(f, x0, opts, bounds) {
  const n = x0.length;
  const memory = opts.memory ?? DEFAULT_MEMORY;
  const maxIter = opts.maxIter ?? DEFAULT_MAX_ITER;
  const gtol = opts.gtol ?? DEFAULT_GTOL;
  const ftol = opts.ftol ?? DEFAULT_FTOL;
  const step = opts.step ?? DEFAULT_FD_STEP;
  const gradient = opts.gradient ?? ((x) => numericGradient(f, x, step));

  const projectGrad = (grad, at) => {
    if (!bounds) return grad;
    return grad.map((gj, j) => {
      const b = bounds[j];
      if (!b) return gj;
      const [lo, hi] = b;
      if (at[j] <= lo && gj > 0) return 0;
      if (at[j] >= hi && gj < 0) return 0;
      return gj;
    });
  };

  let x = clampToBounds(x0, bounds);
  let fx = f(x);
  let g = gradient(x);
  let pg = projectGrad(g, x);
  const sList = [];
  const yList = [];
  const rhoList = [];

  let iterations = 0;
  for (; iterations < maxIter; iterations++) {
    if (norm(pg) < gtol) return { point: x, value: fx, iterations, converged: true };
    const dir = scale(twoLoop(pg, sList, yList, rhoList), -1);
    if (dot(dir, pg) >= 0) {
      for (let j = 0; j < n; j++) dir[j] = -pg[j];
    }

    let t = 1;
    let xNew = x;
    let fNew = fx;
    let accepted = false;
    let clamped = false;
    const slope = dot(pg, dir);
    for (let ls = 0; ls < MAX_LINE_SEARCH; ls++) {
      const raw = x.map((v, j) => v + t * dir[j]);
      const cand = clampToBounds(raw, bounds);
      const fc = f(cand);
      if (fc <= fx + WOLFE_C1 * t * slope) {
        xNew = cand;
        fNew = fc;
        accepted = true;
        clamped = bounds != null && cand.some((v, j) => v !== raw[j]);
        break;
      }
      const denom = 2 * (fc - fx - slope * t);
      const tq = denom > 0 ? (-slope * t * t) / denom : t * BACKTRACK_MAX;
      const lo = BACKTRACK_MIN * t;
      const hi = BACKTRACK_MAX * t;
      t = tq < lo ? lo : tq > hi ? hi : tq;
    }
    if (!accepted) return { point: x, value: fx, iterations, converged: false };

    const gNew = gradient(xNew);
    const pgNew = projectGrad(gNew, xNew);
    const s = sub(xNew, x);
    const y = clamped ? sub(pgNew, pg) : sub(gNew, g);
    const sy = dot(s, y);
    if (sy > Number.EPSILON * norm(s) * norm(y)) {
      sList.push(s);
      yList.push(y);
      rhoList.push(1 / sy);
      if (sList.length > memory) {
        sList.shift();
        yList.shift();
        rhoList.shift();
      }
    }
    const fDelta = Math.abs(fNew - fx);
    x = xNew;
    fx = fNew;
    g = gNew;
    pg = pgNew;
    if (fDelta < ftol * (1 + Math.abs(fx)) && norm(s) < ftol * (1 + norm(x))) {
      iterations++;
      return { point: x, value: fx, iterations, converged: true };
    }
  }
  return { point: x, value: fx, iterations, converged: norm(pg) < gtol };
}

export function lbfgs(f, x0, opts = {}) {
  return minimize(f, x0, opts, null);
}

export function lbfgsB(f, x0, bounds, opts = {}) {
  return minimize(f, x0, opts, bounds);
}
