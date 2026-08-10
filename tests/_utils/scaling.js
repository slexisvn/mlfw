function timeOnce(work, input) {
  const t0 = performance.now();
  work(input);
  return performance.now() - t0;
}

export function scalingRatio({ build, work, n, runs = 5, warmups = 2 }) {
  const small = build(n);
  const large = build(2 * n);
  for (let i = 0; i < warmups; i++) {
    work(small);
    work(large);
  }
  let tSmall = Infinity;
  let tLarge = Infinity;
  for (let i = 0; i < runs; i++) {
    tSmall = Math.min(tSmall, timeOnce(work, small));
    tLarge = Math.min(tLarge, timeOnce(work, large));
  }
  return { ratio: tLarge / Math.max(tSmall, 1e-3), tSmall, tLarge };
}

export const QUADRATIC_RATIO = 3;
