function timeOnce(work, input) {
  const t0 = performance.now();
  work(input);
  return performance.now() - t0;
}

function measure(build, work, n, runs, warmups) {
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
  return { tSmall, tLarge };
}

export function scalingRatio({ build, work, n, runs = 5, warmups = 2, minMs = 5, maxGrowth = 4 }) {
  let size = n;
  let m = measure(build, work, size, runs, warmups);
  for (let i = 0; i < maxGrowth && m.tSmall < minMs; i++) {
    size *= 2;
    m = measure(build, work, size, runs, warmups);
  }
  return { ratio: m.tLarge / m.tSmall, tSmall: m.tSmall, tLarge: m.tLarge, n: size };
}

export const QUADRATIC_RATIO = 3;
