export type Factorization = number[];

function divisorsOf(n: number): number[] {
  const out: number[] = [];
  for (let d = 1; d * d <= n; d++) {
    if (n % d !== 0) continue;
    out.push(d);
    const e = n / d;
    if (e !== d) out.push(e);
  }
  out.sort((a, b) => a - b);
  return out;
}

function cmpTuple(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

function selectDiverse(tuples: Factorization[], maxCandidates: number): Factorization[] {
  const seen = new Set<string>();
  const uniq: Factorization[] = [];
  tuples.sort(cmpTuple);
  for (const t of tuples) {
    const k = t.join(',');
    if (seen.has(k)) continue;
    seen.add(k);
    uniq.push(t);
  }
  if (uniq.length <= maxCandidates) return uniq;
  const picked = new Set<number>();
  const out: Factorization[] = [];
  for (let i = 0; i < maxCandidates; i++) {
    const idx = Math.floor((i * (uniq.length - 1)) / (maxCandidates - 1));
    if (picked.has(idx)) continue;
    picked.add(idx);
    out.push(uniq[idx]);
  }
  return out;
}

export function enumerateFactorizations(extent: number, levels: number, maxCandidates = 48): Factorization[] {
  if (!Number.isFinite(extent) || extent < 1 || levels <= 1) return [[extent]];
  const all: Factorization[] = [];
  const limit = maxCandidates * 8;
  const rec = (remaining: number, depth: number, acc: number[]): void => {
    if (all.length >= limit) return;
    if (depth === levels - 1) {
      all.push([...acc, remaining]);
      return;
    }
    for (const d of divisorsOf(remaining)) {
      acc.push(d);
      rec(remaining / d, depth + 1, acc);
      acc.pop();
      if (all.length >= limit) return;
    }
  };
  rec(extent, 0, []);
  return selectDiverse(all, maxCandidates);
}
