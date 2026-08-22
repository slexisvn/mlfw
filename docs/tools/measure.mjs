// Shared timing helper for the book's labs.
//
// Every lab that reports a duration reports it the same way, for the reason
// given in Chapter 1 §1.8: the minimum of n runs is not a robust statistic.
// It estimates the machine's best case, it is biased downward, it drifts
// further from the truth as n grows, and it discards the dispersion that tells
// a reader whether a ratio is real or noise. So the headline number here is the
// median, and the spread travels with it.

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function summarize(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const q1 = quantile(sorted, 0.25);
  const q3 = quantile(sorted, 0.75);
  const median = quantile(sorted, 0.5);
  return {
    n,
    median,
    min: sorted[0],
    max: sorted[n - 1],
    q1,
    q3,
    iqr: q3 - q1,
    // Relative half-IQR: a scale-free measure of how noisy the sample is.
    // Ratios smaller than this are not distinguishable from noise.
    rsd: median === 0 ? 0 : (q3 - q1) / 2 / median,
  };
}

// Run `fn` `reps` times after `warmup` untimed calls, and summarize.
// `fn` may be sync or async; the timer wraps the awaited call either way.
export async function measure(fn, { reps = 30, warmup = 5 } = {}) {
  for (let i = 0; i < warmup; i++) await fn();
  const samples = new Array(reps);
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await fn();
    samples[i] = performance.now() - t0;
  }
  return summarize(samples);
}

// "12.340 ms  (n=30, IQR 11.900-12.900, min 11.700, max 19.400)"
export function format(stat, unit = 'ms', digits = 3) {
  const f = (x) => x.toFixed(digits);
  return `${f(stat.median)} ${unit}  (n=${stat.n}, IQR ${f(stat.q1)}-${f(stat.q3)}, min ${f(stat.min)}, max ${f(stat.max)})`;
}

// A speedup is only worth reporting if it is larger than the noise in the two
// samples it came from. This returns the ratio of medians together with a flag
// saying whether the two IQRs overlap, in which case the ratio means nothing.
export function ratio(baseline, candidate) {
  const value = baseline.median / candidate.median;
  const overlapping = baseline.q1 <= candidate.q3 && candidate.q1 <= baseline.q3;
  return { value, overlapping, note: overlapping ? 'IQRs overlap - inside the noise' : '' };
}

export function formatRatio(baseline, candidate, digits = 2) {
  const r = ratio(baseline, candidate);
  return `${r.value.toFixed(digits)}x${r.note ? `  [${r.note}]` : ''}`;
}
