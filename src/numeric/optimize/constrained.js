import { lbfgs } from './lbfgs.js';
import { clampToBounds } from './_util.js';

const DEFAULT_PENALTY = 10;
const DEFAULT_PENALTY_GROWTH = 10;
const DEFAULT_OUTER_ITER = 20;
const DEFAULT_CTOL = 1e-8;

export function constrainedMinimize(f, x0, opts = {}) {
  const bounds = opts.bounds ?? null;
  const inequalities = opts.inequalities ?? [];
  const equalities = opts.equalities ?? [];
  const outerIter = opts.outerIter ?? DEFAULT_OUTER_ITER;
  const ctol = opts.ctol ?? DEFAULT_CTOL;
  const growth = opts.penaltyGrowth ?? DEFAULT_PENALTY_GROWTH;
  const inner = opts.inner ?? lbfgs;
  const innerOpts = opts.innerOpts ?? {};

  let mu = opts.penalty ?? DEFAULT_PENALTY;
  let x = clampToBounds(x0, bounds);
  let totalIterations = 0;
  let last = null;

  for (let outer = 0; outer < outerIter; outer++) {
    const penalized = (p) => {
      const xp = clampToBounds(p, bounds);
      let val = f(xp);
      for (const g of inequalities) {
        const v = g(xp);
        if (v > 0) val += mu * v * v;
      }
      for (const h of equalities) {
        const v = h(xp);
        val += mu * v * v;
      }
      return val;
    };
    last = inner(penalized, x, innerOpts);
    x = clampToBounds(last.point, bounds);
    totalIterations += last.iterations;

    let violation = 0;
    for (const g of inequalities) violation = Math.max(violation, g(x));
    for (const h of equalities) violation = Math.max(violation, Math.abs(h(x)));
    if (violation < ctol) {
      return { point: x, value: f(x), iterations: totalIterations, converged: true };
    }
    mu *= growth;
  }
  return { point: x, value: f(x), iterations: totalIterations, converged: false };
}
