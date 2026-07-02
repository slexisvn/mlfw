import { wrapResult, tensorToContiguous } from '../../../dispatcher/jit_dispatch.js';
import { DEFAULT_RCOND } from '../../cpu/linalg/config.js';
import { csGesvd, csSyevd, csPotrf, csGetrf, csSolveLU } from '../../../runtime/node/cuda/cusolver.js';
import { cublasGemmDevice } from '../../../runtime/node/cuda/cublas.js';
import { acquire, release, copyHostToDevice, copyDeviceToHost } from '../../../runtime/node/cuda/memory.js';
import { cu, checkCU } from '../../../runtime/node/cuda/ffi.js';
import { getDevice } from '../../../runtime/node/cuda/device.js';

const OP_T = 1;

function hostData(t) {
  return Float32Array.from(tensorToContiguous(t));
}

function require2D(t, op) {
  if (t.ndim !== 2) throw new Error(`linalg.${op}: expected a 2-D matrix, got ${t.ndim}-D`);
}

function requireF32(t, op) {
  if (t.dtype !== 'f32') throw new Error(`linalg.${op}: GPU backend supports f32 tensors only, got '${t.dtype}'`);
}

function requireSquare(t, op) {
  if (t.shape[0] !== t.shape[1]) throw new Error(`linalg.${op}: matrix must be square`);
}

function syncStream() {
  checkCU('cuStreamSynchronize', cu.streamSynchronize(getDevice().stream));
}

function _svdRowMajor(a, m, n) {
  const k = Math.min(m, n);
  const U = new Float32Array(m * k), S = new Float32Array(k), V = new Float32Array(n * k);
  if (m >= n) {
    const Acm = new Float32Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Acm[i + j * m] = a[i * n + j];
    const r = csGesvd(Acm, m, n);
    for (let c = 0; c < k; c++) {
      S[c] = r.S[c];
      for (let i = 0; i < m; i++) U[i * k + c] = r.U[i + c * m];
      for (let j = 0; j < n; j++) V[j * k + c] = r.VT[c + j * n];
    }
  } else {
    const r = csGesvd(a, n, m);
    for (let c = 0; c < k; c++) {
      S[c] = r.S[c];
      for (let i = 0; i < m; i++) U[i * k + c] = r.VT[c + i * m];
      for (let j = 0; j < n; j++) V[j * k + c] = r.U[j + c * n];
    }
  }
  return { U, S, V, k };
}

export function gpuSvd(_ks, a) {
  require2D(a, 'svd'); requireF32(a, 'svd');
  const [m, n] = a.shape;
  const { U, S, V, k } = _svdRowMajor(hostData(a), m, n);
  return [
    wrapResult(U, [m, k], a.dtype, a.device),
    wrapResult(S, [k], a.dtype, a.device),
    wrapResult(V, [n, k], a.dtype, a.device),
  ];
}

export function gpuEigh(_ks, a) {
  require2D(a, 'eigh'); requireF32(a, 'eigh'); requireSquare(a, 'eigh');
  const n = a.shape[0];
  const { W, V } = csSyevd(hostData(a), n);
  const vectors = new Float32Array(n * n);
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) vectors[row * n + col] = V[row + col * n];
  return [wrapResult(W, [n], a.dtype, a.device), wrapResult(vectors, [n, n], a.dtype, a.device)];
}

export function gpuCholesky(_ks, a) {
  require2D(a, 'cholesky'); requireF32(a, 'cholesky'); requireSquare(a, 'cholesky');
  const n = a.shape[0];
  const Lcm = csPotrf(hostData(a), n);
  const L = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * n + j] = Lcm[i + j * n];
  return wrapResult(L, [n, n], a.dtype, a.device);
}

export function gpuSolve(_ks, a, b) {
  require2D(a, 'solve'); requireF32(a, 'solve'); requireSquare(a, 'solve'); requireF32(b, 'solve');
  const n = a.shape[0];
  const bVec = b.ndim === 1;
  const nrhs = bVec ? 1 : b.shape[1];
  if (b.shape[0] !== n) throw new Error('linalg.solve: right-hand side rows must match matrix');
  const bData = hostData(b);
  const Bcm = new Float32Array(n * nrhs);
  for (let i = 0; i < n; i++) for (let r = 0; r < nrhs; r++) Bcm[i + r * n] = bData[i * nrhs + r];
  const X = csSolveLU(hostData(a), n, Bcm, nrhs, OP_T);
  const out = new Float32Array(n * nrhs);
  for (let i = 0; i < n; i++) for (let r = 0; r < nrhs; r++) out[i * nrhs + r] = X[i + r * n];
  return wrapResult(out, bVec ? [n] : [n, nrhs], a.dtype, a.device);
}

export function gpuInv(_ks, a) {
  require2D(a, 'inv'); requireF32(a, 'inv'); requireSquare(a, 'inv');
  const n = a.shape[0];
  const I = new Float32Array(n * n);
  for (let i = 0; i < n; i++) I[i + i * n] = 1;
  const X = csSolveLU(hostData(a), n, I, n, OP_T);
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = X[i + j * n];
  return wrapResult(out, [n, n], a.dtype, a.device);
}

export function gpuDet(_ks, a) {
  require2D(a, 'det'); requireF32(a, 'det'); requireSquare(a, 'det');
  const n = a.shape[0];
  const { LU, ipiv } = csGetrf(hostData(a), n);
  let sign = 1, prod = 1;
  for (let i = 0; i < n; i++) {
    prod *= LU[i + i * n];
    if (ipiv[i] !== i + 1) sign = -sign;
  }
  return wrapResult(new Float32Array([sign * prod]), [], a.dtype, a.device);
}

export function gpuPinv(_ks, a) {
  require2D(a, 'pinv'); requireF32(a, 'pinv');
  const [m, n] = a.shape;
  const { U, S, V, k } = _svdRowMajor(hostData(a), m, n);
  const cutoff = DEFAULT_RCOND * (k ? S[0] : 0);
  const out = new Float32Array(n * m);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let c = 0; c < k; c++) { const sv = S[c]; if (sv > cutoff) s += V[j * k + c] * (1 / sv) * U[i * k + c]; }
      out[j * m + i] = s;
    }
  }
  return wrapResult(out, [n, m], a.dtype, a.device);
}

export function gpuLstsq(_ks, a, b) {
  require2D(a, 'lstsq'); requireF32(a, 'lstsq'); requireF32(b, 'lstsq');
  const [m, n] = a.shape;
  const bVec = b.ndim === 1;
  const nrhs = bVec ? 1 : b.shape[1];
  if (b.shape[0] !== m) throw new Error('linalg.lstsq: right-hand side rows must match matrix');
  const bData = hostData(b);
  const { U, S, V, k } = _svdRowMajor(hostData(a), m, n);
  const cutoff = DEFAULT_RCOND * (k ? S[0] : 0);
  const t = new Float32Array(k * nrhs);
  for (let c = 0; c < k; c++) {
    const sv = S[c];
    for (let r = 0; r < nrhs; r++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += U[i * k + c] * bData[i * nrhs + r];
      t[c * nrhs + r] = sv > cutoff ? s / sv : 0;
    }
  }
  const x = new Float32Array(n * nrhs);
  for (let j = 0; j < n; j++) {
    for (let r = 0; r < nrhs; r++) {
      let s = 0;
      for (let c = 0; c < k; c++) s += V[j * k + c] * t[c * nrhs + r];
      x[j * nrhs + r] = s;
    }
  }
  return wrapResult(x, bVec ? [n] : [n, nrhs], a.dtype, a.device);
}

export function gpuCov(_ks, a) {
  require2D(a, 'cov'); requireF32(a, 'cov');
  const [rows, cols] = a.shape;
  const data = hostData(a);
  const mean = new Float32Array(cols);
  for (let j = 0; j < cols; j++) { let s = 0; for (let i = 0; i < rows; i++) s += data[i * cols + j]; mean[j] = s / rows; }
  const Xc = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) Xc[i * cols + j] = data[i * cols + j] - mean[j];
  const denom = rows > 1 ? rows - 1 : 1;
  getDevice();
  const dXc = acquire(Xc.byteLength);
  const outArr = new Float32Array(cols * cols);
  const dC = acquire(outArr.byteLength);
  try {
    copyHostToDevice(dXc, Xc);
    cublasGemmDevice(cols, cols, rows, dXc, true, dXc, false, dC);
    syncStream();
    copyDeviceToHost(outArr, dC);
  } finally {
    release(dXc, Xc.byteLength);
    release(dC, outArr.byteLength);
  }
  for (let i = 0; i < cols * cols; i++) outArr[i] /= denom;
  return wrapResult(outArr, [cols, cols], a.dtype, a.device);
}
