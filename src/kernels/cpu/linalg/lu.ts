export type LuFactorization = { lu: Float64Array; piv: Int32Array; sign: number };

export function luFactor(a: Float64Array, n: number): LuFactorization {
  const lu = Float64Array.from(a);
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  let sign = 1;
  for (let k = 0; k < n; k++) {
    let p = k;
    let max = Math.abs(lu[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i * n + k]);
      if (v > max) { max = v; p = i; }
    }
    if (max === 0) throw new Error('linalg: matrix is singular');
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const tmp = lu[k * n + j];
        lu[k * n + j] = lu[p * n + j];
        lu[p * n + j] = tmp;
      }
      const tp = piv[k];
      piv[k] = piv[p];
      piv[p] = tp;
      sign = -sign;
    }
    const diag = lu[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = lu[i * n + k] / diag;
      lu[i * n + k] = f;
      for (let j = k + 1; j < n; j++) lu[i * n + j] -= f * lu[k * n + j];
    }
  }
  return { lu, piv, sign };
}

export function luSolve(lu: Float64Array, piv: Int32Array, n: number, b: Float64Array, nrhs: number): Float64Array {
  const x = new Float64Array(n * nrhs);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nrhs; c++) x[i * nrhs + c] = b[piv[i] * nrhs + c];
  }
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nrhs; c++) {
      let s = x[i * nrhs + c];
      for (let j = 0; j < i; j++) s -= lu[i * n + j] * x[j * nrhs + c];
      x[i * nrhs + c] = s;
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    const diag = lu[i * n + i];
    for (let c = 0; c < nrhs; c++) {
      let s = x[i * nrhs + c];
      for (let j = i + 1; j < n; j++) s -= lu[i * n + j] * x[j * nrhs + c];
      x[i * nrhs + c] = s / diag;
    }
  }
  return x;
}

export function solveHost(a: Float64Array, n: number, b: Float64Array, nrhs: number): Float64Array {
  const { lu, piv } = luFactor(a, n);
  return luSolve(lu, piv, n, b, nrhs);
}

export function detHost(a: Float64Array, n: number): number {
  let factored: LuFactorization;
  try {
    factored = luFactor(a, n);
  } catch {
    return 0;
  }
  let d = factored.sign;
  for (let i = 0; i < n; i++) d *= factored.lu[i * n + i];
  return d;
}
