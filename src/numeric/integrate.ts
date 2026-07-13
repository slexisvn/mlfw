const DEFAULT_N = 1024;
const DEFAULT_QUAD_TOL = 1e-10;
const DEFAULT_MAX_DEPTH = 50;
const RICHARDSON = 15;

type IntegrateOptions = { n?: number; tol?: number; maxDepth?: number };
type ScalarFn = (x: number) => number;

export function trapezoid(f: ScalarFn, a: number, b: number, opts: IntegrateOptions = {}): number {
  const n = opts.n ?? DEFAULT_N;
  const h = (b - a) / n;
  let s = 0.5 * (f(a) + f(b));
  for (let i = 1; i < n; i++) s += f(a + i * h);
  return s * h;
}

export function simpson(f: ScalarFn, a: number, b: number, opts: IntegrateOptions = {}): number {
  let n = opts.n ?? DEFAULT_N;
  if (n % 2 === 1) n += 1;
  const h = (b - a) / n;
  let s = f(a) + f(b);
  for (let i = 1; i < n; i++) s += (i % 2 === 0 ? 2 : 4) * f(a + i * h);
  return (s * h) / 3;
}

function simpsonCell(fa: number, fb: number, fm: number, a: number, b: number): number {
  return ((b - a) / 6) * (fa + 4 * fm + fb);
}

function adaptiveStep(f: ScalarFn, a: number, b: number, fa: number, fb: number, fm: number, whole: number, tol: number, depth: number, maxDepth: number): number {
  const m = 0.5 * (a + b);
  const lm = 0.5 * (a + m);
  const rm = 0.5 * (m + b);
  const flm = f(lm);
  const frm = f(rm);
  const left = simpsonCell(fa, fm, flm, a, m);
  const right = simpsonCell(fm, fb, frm, m, b);
  const delta = left + right - whole;
  if (depth >= maxDepth || Math.abs(delta) <= RICHARDSON * tol) {
    return left + right + delta / RICHARDSON;
  }
  return adaptiveStep(f, a, m, fa, fm, flm, left, 0.5 * tol, depth + 1, maxDepth)
    + adaptiveStep(f, m, b, fm, fb, frm, right, 0.5 * tol, depth + 1, maxDepth);
}

export function quadrature(f: ScalarFn, a: number, b: number, opts: IntegrateOptions = {}): number {
  const tol = opts.tol ?? DEFAULT_QUAD_TOL;
  const maxDepth = opts.maxDepth ?? DEFAULT_MAX_DEPTH;
  const m = 0.5 * (a + b);
  const fa = f(a);
  const fb = f(b);
  const fm = f(m);
  const whole = simpsonCell(fa, fb, fm, a, b);
  return adaptiveStep(f, a, b, fa, fb, fm, whole, tol, 0, maxDepth);
}
