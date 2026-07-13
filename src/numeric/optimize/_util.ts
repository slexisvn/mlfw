import type { Bounds, VectorFn } from '../types.js';

export const DEFAULT_FD_STEP = 1e-7;

export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}

export function add(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function sub(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function scale(a: readonly number[], s: number): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
  return out;
}

export function argmin(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[best]) best = i;
  return best;
}

export function numericGradient(f: VectorFn, x: readonly number[], step: number): number[] {
  const n = x.length;
  const g = new Array<number>(n);
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

export function clampToBounds(x: readonly number[], bounds?: Bounds | null): number[] {
  if (!bounds) return x.slice();
  return x.map((v, j) => {
    const b = bounds[j];
    if (!b) return v;
    const [lo, hi] = b;
    return v < lo ? lo : v > hi ? hi : v;
  });
}
