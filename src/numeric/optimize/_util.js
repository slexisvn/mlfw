export const DEFAULT_FD_STEP = 1e-7;

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function add(a, b) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function sub(a, b) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function scale(a, s) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
  return out;
}

export function argmin(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[best]) best = i;
  return best;
}

export function numericGradient(f, x, step) {
  const n = x.length;
  const g = new Array(n);
  const p = x.slice();
  for (let i = 0; i < n; i++) {
    const xi = p[i];
    p[i] = xi + step;
    const fp = f(p);
    p[i] = xi - step;
    const fm = f(p);
    p[i] = xi;
    g[i] = (fp - fm) / (2 * step);
  }
  return g;
}

export function clampToBounds(x, bounds) {
  if (!bounds) return x.slice();
  return x.map((v, j) => {
    const b = bounds[j];
    if (!b) return v;
    const [lo, hi] = b;
    return v < lo ? lo : v > hi ? hi : v;
  });
}
