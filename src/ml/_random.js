const DEFAULT_SEED = 0x9e3779b9;

export function makeRng(seed) {
  let a = (seed ?? DEFAULT_SEED) >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randInt(rng, n) {
  return Math.floor(rng() * n);
}

export function shuffledIndices(n, rng) {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}
