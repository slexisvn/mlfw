const DEFAULT_TOL = 1e-12;
const DEFAULT_MAX_ITER = 100;
const DEFAULT_FD_STEP = 1e-7;

type RootOptions = { tol?: number; maxIter?: number; step?: number; derivative?: (x: number) => number };
type RootResult = { root: number; iterations: number; converged: boolean };

function result(root: number, iterations: number, converged: boolean): RootResult {
  return { root, iterations, converged };
}

export function bisect(f: (x: number) => number, a: number, b: number, opts: RootOptions = {}): RootResult {
  const tol = opts.tol ?? DEFAULT_TOL;
  const maxIter = opts.maxIter ?? DEFAULT_MAX_ITER;
  let fa = f(a);
  let fb = f(b);
  if (fa === 0) return result(a, 0, true);
  if (fb === 0) return result(b, 0, true);
  if (fa * fb > 0) throw new Error('bisect: f(a) and f(b) must have opposite signs');
  let lo = a;
  let hi = b;
  let flo = fa;
  for (let i = 1; i <= maxIter; i++) {
    const mid = 0.5 * (lo + hi);
    const fmid = f(mid);
    if (fmid === 0 || 0.5 * (hi - lo) < tol) return result(mid, i, true);
    if (flo * fmid < 0) {
      hi = mid;
    } else {
      lo = mid;
      flo = fmid;
    }
  }
  return result(0.5 * (lo + hi), maxIter, false);
}

export function newton(f: (x: number) => number, x0: number, opts: RootOptions = {}): RootResult {
  const tol = opts.tol ?? DEFAULT_TOL;
  const maxIter = opts.maxIter ?? DEFAULT_MAX_ITER;
  const step = opts.step ?? DEFAULT_FD_STEP;
  const df = opts.derivative ?? ((x: number) => (f(x + step) - f(x - step)) / (2 * step));
  let x = x0;
  for (let i = 1; i <= maxIter; i++) {
    const fx = f(x);
    if (Math.abs(fx) < tol) return result(x, i, true);
    const d = df(x);
    if (d === 0) break;
    const xn = x - fx / d;
    if (Math.abs(xn - x) < tol) return result(xn, i, true);
    x = xn;
  }
  return result(x, maxIter, false);
}

export function brentq(f: (x: number) => number, a: number, b: number, opts: RootOptions = {}): RootResult {
  const tol = opts.tol ?? DEFAULT_TOL;
  const maxIter = opts.maxIter ?? DEFAULT_MAX_ITER;
  let fa = f(a);
  let fb = f(b);
  if (fa === 0) return result(a, 0, true);
  if (fb === 0) return result(b, 0, true);
  if (fa * fb > 0) throw new Error('brentq: f(a) and f(b) must have opposite signs');
  if (Math.abs(fa) < Math.abs(fb)) {
    [a, b] = [b, a];
    [fa, fb] = [fb, fa];
  }
  let c = a;
  let fc = fa;
  let d = c;
  let mflag = true;
  for (let i = 1; i <= maxIter; i++) {
    let s;
    if (fa !== fc && fb !== fc) {
      s = (a * fb * fc) / ((fa - fb) * (fa - fc))
        + (b * fa * fc) / ((fb - fa) * (fb - fc))
        + (c * fa * fb) / ((fc - fa) * (fc - fb));
    } else {
      s = b - fb * (b - a) / (fb - fa);
    }
    const bound = 0.25 * (3 * a + b);
    const outside = !(s > Math.min(bound, b) && s < Math.max(bound, b));
    const slow1 = mflag && Math.abs(s - b) >= 0.5 * Math.abs(b - c);
    const slow2 = !mflag && Math.abs(s - b) >= 0.5 * Math.abs(c - d);
    const tiny1 = mflag && Math.abs(b - c) < tol;
    const tiny2 = !mflag && Math.abs(c - d) < tol;
    if (outside || slow1 || slow2 || tiny1 || tiny2) {
      s = 0.5 * (a + b);
      mflag = true;
    } else {
      mflag = false;
    }
    const fs = f(s);
    d = c;
    c = b;
    fc = fb;
    if (fa * fs < 0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }
    if (Math.abs(fa) < Math.abs(fb)) {
      [a, b] = [b, a];
      [fa, fb] = [fb, fa];
    }
    if (fs === 0) return result(s, i, true);
    if (Math.abs(fb) < tol || Math.abs(b - a) < tol) return result(b, i, true);
  }
  return result(b, maxIter, false);
}
