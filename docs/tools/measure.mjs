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
    rsd: median === 0 ? 0 : (q3 - q1) / 2 / median,
  };
}

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

export function format(stat, unit = 'ms', digits = 3) {
  const f = (x) => x.toFixed(digits);
  return `${f(stat.median)} ${unit}  (n=${stat.n}, IQR ${f(stat.q1)}-${f(stat.q3)}, min ${f(stat.min)}, max ${f(stat.max)})`;
}

export function ratio(baseline, candidate) {
  const value = baseline.median / candidate.median;
  const overlapping = baseline.q1 <= candidate.q3 && candidate.q1 <= baseline.q3;
  return { value, overlapping, note: overlapping ? 'IQRs overlap - inside the noise' : '' };
}

export function formatRatio(baseline, candidate, digits = 2) {
  const r = ratio(baseline, candidate);
  return `${r.value.toFixed(digits)}x${r.note ? `  [${r.note}]` : ''}`;
}
