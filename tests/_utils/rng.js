export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomArray(seed, n, lo = -1, hi = 1) {
  const r = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = lo + (hi - lo) * r();
  return out;
}
