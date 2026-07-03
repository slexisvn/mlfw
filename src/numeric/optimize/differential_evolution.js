import { makeRng, randInt } from '../../ml/_random.js';
import { argmin } from './_util.js';

const DEFAULT_POP_PER_DIM = 10;
const MIN_POP = 5;
const DEFAULT_MUTATION = 0.8;
const DEFAULT_RECOMBINATION = 0.9;
const DEFAULT_MAX_ITER = 1000;
const DEFAULT_TOL = 1e-10;

function pickDistinct(rng, popSize, exclude) {
  let k;
  do {
    k = randInt(rng, popSize);
  } while (exclude.includes(k));
  return k;
}

export function differentialEvolution(f, bounds, opts = {}) {
  const n = bounds.length;
  const rng = makeRng(opts.seed);
  const popSize = opts.populationSize ?? Math.max(MIN_POP, DEFAULT_POP_PER_DIM * n);
  const mutation = opts.mutation ?? DEFAULT_MUTATION;
  const recombination = opts.recombination ?? DEFAULT_RECOMBINATION;
  const maxIter = opts.maxIter ?? DEFAULT_MAX_ITER;
  const tol = opts.tol ?? DEFAULT_TOL;

  const pop = [];
  for (let i = 0; i < popSize; i++) {
    const ind = new Array(n);
    for (let j = 0; j < n; j++) {
      const [lo, hi] = bounds[j];
      ind[j] = lo + rng() * (hi - lo);
    }
    pop.push(ind);
  }
  const fvals = pop.map(f);
  let best = argmin(fvals);

  let iterations = 0;
  for (; iterations < maxIter; iterations++) {
    for (let i = 0; i < popSize; i++) {
      const a = pickDistinct(rng, popSize, [i]);
      const b = pickDistinct(rng, popSize, [i, a]);
      const c = pickDistinct(rng, popSize, [i, a, b]);
      const forced = randInt(rng, n);
      const trial = pop[i].slice();
      for (let j = 0; j < n; j++) {
        if (rng() < recombination || j === forced) {
          const [lo, hi] = bounds[j];
          let v = pop[a][j] + mutation * (pop[b][j] - pop[c][j]);
          if (v < lo) v = lo;
          else if (v > hi) v = hi;
          trial[j] = v;
        }
      }
      const ft = f(trial);
      if (ft < fvals[i]) {
        pop[i] = trial;
        fvals[i] = ft;
        if (ft < fvals[best]) best = i;
      }
    }
    let lo = fvals[0];
    let hi = fvals[0];
    for (let i = 1; i < popSize; i++) {
      if (fvals[i] < lo) lo = fvals[i];
      if (fvals[i] > hi) hi = fvals[i];
    }
    if (hi - lo < tol) {
      iterations++;
      break;
    }
  }
  return { point: pop[best].slice(), value: fvals[best], iterations, converged: iterations < maxIter };
}
