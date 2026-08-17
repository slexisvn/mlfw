function timeAt(build, work, size, runs, warmups) {
  const input = build(size);
  for (let i = 0; i < warmups; i++) work(input);
  let best = Infinity;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    work(input);
    const dt = performance.now() - t0;
    if (dt < best) best = dt;
  }
  return best;
}

function sweep(build, work, sizes, runs, warmups) {
  const best = new Array(sizes.length).fill(Infinity);
  const ascending = sizes.map((_, i) => i);
  for (const order of [ascending, [...ascending].reverse()]) {
    for (const i of order) {
      const t = timeAt(build, work, sizes[i], runs, warmups);
      if (t < best[i]) best[i] = t;
    }
  }
  return best;
}

function leastSquaresSlope(xs, ys) {
  let xbar = 0, ybar = 0;
  for (let i = 0; i < xs.length; i++) { xbar += xs[i]; ybar += ys[i]; }
  xbar /= xs.length;
  ybar /= ys.length;
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - xbar) * (ys[i] - ybar);
    den += (xs[i] - xbar) * (xs[i] - xbar);
  }
  return num / den;
}

export function scalingExponent({ build, work, n, points = 4, runs = 2, warmups = 1, minMs = 3, maxGrowth = 4 }) {
  let base = n;
  let sizes, times;
  for (let g = 0; ; g++) {
    sizes = Array.from({ length: points }, (_, i) => base * (2 ** i));
    times = sweep(build, work, sizes, runs, warmups);
    if (times[0] >= minMs || g >= maxGrowth) break;
    base *= 2;
  }
  const exponent = leastSquaresSlope(sizes.map(Math.log2), times.map(Math.log2));
  return { exponent, sizes, times };
}

export const SUBQUADRATIC_EXPONENT = 1.85;

export function scalingReport({ exponent, sizes, times }) {
  const measured = sizes.map((s, i) => `${s}:${times[i].toFixed(1)}ms`).join(' ');
  return `${measured} (exponent ${exponent.toFixed(2)}, linear 1.0, quadratic 2.0)`;
}
