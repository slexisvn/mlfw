function upperInterval(xs, x) {
  let lo = 1;
  let hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function evalLinear(xs, ys, x) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  const i = upperInterval(xs, x);
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

export function linearInterp(xs, ys, xq) {
  if (Array.isArray(xq)) return xq.map((x) => evalLinear(xs, ys, x));
  return evalLinear(xs, ys, xq);
}

function thomas(sub, diag, sup, rhs) {
  const n = diag.length;
  const c = new Array(n);
  const d = new Array(n);
  c[0] = sup[0] / diag[0];
  d[0] = rhs[0] / diag[0];
  for (let i = 1; i < n; i++) {
    const denom = diag[i] - sub[i] * c[i - 1];
    c[i] = sup[i] / denom;
    d[i] = (rhs[i] - sub[i] * d[i - 1]) / denom;
  }
  const x = new Array(n);
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
}

export function cubicSpline(xs, ys) {
  const n = xs.length;
  const h = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xs[i + 1] - xs[i];
  const sub = new Array(n).fill(0);
  const diag = new Array(n).fill(1);
  const sup = new Array(n).fill(0);
  const rhs = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    sub[i] = h[i - 1];
    diag[i] = 2 * (h[i - 1] + h[i]);
    sup[i] = h[i];
    rhs[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
  }
  const m = thomas(sub, diag, sup, rhs);
  const evalOne = (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    const i = upperInterval(xs, x);
    const hi = h[i - 1];
    const a = (xs[i] - x) / hi;
    const b = (x - xs[i - 1]) / hi;
    return a * ys[i - 1] + b * ys[i]
      + (((a * a * a - a) * m[i - 1] + (b * b * b - b) * m[i]) * hi * hi) / 6;
  };
  return {
    xs,
    ys,
    coefficients: m,
    evaluate: (xq) => (Array.isArray(xq) ? xq.map(evalOne) : evalOne(xq)),
  };
}
