import koffi from 'koffi';
import { getDevice } from './device.js';
import { setDevice } from './runtime_api.js';
import { cu, checkCU } from './ffi.js';
import type { CudaHandle, DevicePtr } from './ffi.js';
import { acquire, release, copyHostToDevice, copyDeviceToHost } from './memory.js';
import { loadCudaLib, CUSOLVER_SPEC } from './lib_resolver.js';

interface CusolverApi {
  create(h: (CudaHandle | null)[]): number;
  destroy(h: CudaHandle | null): number;
  setStream(h: CudaHandle | null, stream: CudaHandle | null): number;
  gesvdBufferSize(h: CudaHandle | null, m: number, n: number, lwork: number[]): number;
  gesvd(h: CudaHandle | null, jobu: number, jobvt: number, m: number, n: number, A: DevicePtr, lda: number, S: DevicePtr, U: DevicePtr, ldu: number, VT: DevicePtr, ldvt: number, work: DevicePtr, lwork: number, rwork: DevicePtr, info: DevicePtr): number;
  syevdBufferSize(h: CudaHandle | null, jobz: number, uplo: number, n: number, A: DevicePtr, lda: number, W: DevicePtr, lwork: number[]): number;
  syevd(h: CudaHandle | null, jobz: number, uplo: number, n: number, A: DevicePtr, lda: number, W: DevicePtr, work: DevicePtr, lwork: number, info: DevicePtr): number;
  potrfBufferSize(h: CudaHandle | null, uplo: number, n: number, A: DevicePtr, lda: number, lwork: number[]): number;
  potrf(h: CudaHandle | null, uplo: number, n: number, A: DevicePtr, lda: number, work: DevicePtr, lwork: number, info: DevicePtr): number;
  getrfBufferSize(h: CudaHandle | null, m: number, n: number, A: DevicePtr, lda: number, lwork: number[]): number;
  getrf(h: CudaHandle | null, m: number, n: number, A: DevicePtr, lda: number, work: DevicePtr, ipiv: DevicePtr, info: DevicePtr): number;
  getrs(h: CudaHandle | null, trans: number, n: number, nrhs: number, A: DevicePtr, lda: number, ipiv: DevicePtr, B: DevicePtr, ldb: number, info: DevicePtr): number;
}

type CusolverScope = {
  up(hostArr: ArrayBufferView): DevicePtr;
  out(bytes: number): DevicePtr;
  free(): void;
};

let c: CusolverApi, _inited: boolean | null = null;
function ensure(): boolean {
  if (_inited !== null) return _inited;
  try {
    const sol = koffi.load(loadCudaLib(CUSOLVER_SPEC));
    c = {
      create: sol.func('int cusolverDnCreate(_Out_ void **h)'),
      destroy: sol.func('int cusolverDnDestroy(void *h)'),
      setStream: sol.func('int cusolverDnSetStream(void *h, void *stream)'),
      gesvdBufferSize: sol.func('int cusolverDnSgesvd_bufferSize(void *h, int m, int n, _Out_ int *lwork)'),
      gesvd: sol.func('int cusolverDnSgesvd(void *h, int8 jobu, int8 jobvt, int m, int n, uint64 A, int lda, uint64 S, uint64 U, int ldu, uint64 VT, int ldvt, uint64 work, int lwork, uint64 rwork, uint64 info)'),
      syevdBufferSize: sol.func('int cusolverDnSsyevd_bufferSize(void *h, int jobz, int uplo, int n, uint64 A, int lda, uint64 W, _Out_ int *lwork)'),
      syevd: sol.func('int cusolverDnSsyevd(void *h, int jobz, int uplo, int n, uint64 A, int lda, uint64 W, uint64 work, int lwork, uint64 info)'),
      potrfBufferSize: sol.func('int cusolverDnSpotrf_bufferSize(void *h, int uplo, int n, uint64 A, int lda, _Out_ int *lwork)'),
      potrf: sol.func('int cusolverDnSpotrf(void *h, int uplo, int n, uint64 A, int lda, uint64 work, int lwork, uint64 info)'),
      getrfBufferSize: sol.func('int cusolverDnSgetrf_bufferSize(void *h, int m, int n, uint64 A, int lda, _Out_ int *lwork)'),
      getrf: sol.func('int cusolverDnSgetrf(void *h, int m, int n, uint64 A, int lda, uint64 work, uint64 ipiv, uint64 info)'),
      getrs: sol.func('int cusolverDnSgetrs(void *h, int trans, int n, int nrhs, uint64 A, int lda, uint64 ipiv, uint64 B, int ldb, uint64 info)'),
    };
    _inited = true;
  } catch (e) {
    _inited = false;
  }
  return _inited;
}

export function cusolverAvailable(): boolean { return ensure(); }

const JOB_ECON = 'S'.charCodeAt(0);
const EIG_VECTOR = 1, FILL_LOWER = 0;

let _handle: CudaHandle | null = null;
function handle(): CudaHandle | null {
  if (!ensure()) throw new Error('cuSOLVER library not found; install the CUDA Toolkit (cusolver) to use GPU linalg');
  const dev = getDevice();
  setDevice();
  if (!_handle) {
    const h: (CudaHandle | null)[] = [null];
    ck('create', c.create(h));
    _handle = h[0];
    ck('setStream', c.setStream(_handle, dev.stream));
  }
  return _handle;
}

function ck(label: string, status: number): void {
  if (status !== 0) throw new Error('cuSOLVER ' + label + ' failed: ' + status);
}

function sync(): void {
  checkCU('cuStreamSynchronize', cu.streamSynchronize(getDevice().stream));
}

function scope(): CusolverScope {
  const allocs: [DevicePtr, number][] = [];
  return {
    up(hostArr) {
      const p = acquire(hostArr.byteLength);
      copyHostToDevice(p, hostArr);
      allocs.push([p, hostArr.byteLength]);
      return p;
    },
    out(bytes) {
      const b = Math.max(bytes, 1);
      const p = acquire(b);
      allocs.push([p, b]);
      return p;
    },
    free() { for (const [p, b] of allocs) release(p, b); },
  };
}

function readInt(dptr: DevicePtr): number {
  const h = new Int32Array(1);
  copyDeviceToHost(h, dptr);
  return h[0];
}

export function csGesvd(Acm: ArrayBufferView, m: number, n: number): { S: Float32Array; U: Float32Array; VT: Float32Array } {
  const h = handle();
  const s = scope();
  try {
    const dA = s.up(Acm);
    const lwork = [0];
    ck('gesvd_bufferSize', c.gesvdBufferSize(h, m, n, lwork));
    const dS = s.out(n * 4), dU = s.out(m * n * 4), dVT = s.out(n * n * 4);
    const dWork = s.out(lwork[0] * 4), dInfo = s.out(4);
    ck('gesvd', c.gesvd(h, JOB_ECON, JOB_ECON, m, n, dA, m, dS, dU, m, dVT, n, dWork, lwork[0], 0n, dInfo));
    sync();
    const info = readInt(dInfo);
    if (info < 0) throw new Error('cuSOLVER gesvd: invalid argument ' + (-info));
    if (info > 0) throw new Error('cuSOLVER gesvd: did not converge');
    const S = new Float32Array(n), U = new Float32Array(m * n), VT = new Float32Array(n * n);
    copyDeviceToHost(S, dS); copyDeviceToHost(U, dU); copyDeviceToHost(VT, dVT);
    return { S, U, VT };
  } finally { s.free(); }
}

export function csSyevd(Acm: ArrayBufferView, n: number): { W: Float32Array; V: Float32Array } {
  const h = handle();
  const s = scope();
  try {
    const dA = s.up(Acm), dW = s.out(n * 4);
    const lwork = [0];
    ck('syevd_bufferSize', c.syevdBufferSize(h, EIG_VECTOR, FILL_LOWER, n, dA, n, dW, lwork));
    const dWork = s.out(lwork[0] * 4), dInfo = s.out(4);
    ck('syevd', c.syevd(h, EIG_VECTOR, FILL_LOWER, n, dA, n, dW, dWork, lwork[0], dInfo));
    sync();
    const info = readInt(dInfo);
    if (info !== 0) throw new Error('cuSOLVER syevd: failed to converge (info ' + info + ')');
    const W = new Float32Array(n), V = new Float32Array(n * n);
    copyDeviceToHost(W, dW); copyDeviceToHost(V, dA);
    return { W, V };
  } finally { s.free(); }
}

export function csPotrf(Acm: ArrayBufferView, n: number): Float32Array {
  const h = handle();
  const s = scope();
  try {
    const dA = s.up(Acm);
    const lwork = [0];
    ck('potrf_bufferSize', c.potrfBufferSize(h, FILL_LOWER, n, dA, n, lwork));
    const dWork = s.out(lwork[0] * 4), dInfo = s.out(4);
    ck('potrf', c.potrf(h, FILL_LOWER, n, dA, n, dWork, lwork[0], dInfo));
    sync();
    const info = readInt(dInfo);
    if (info > 0) throw new Error('linalg.cholesky: matrix is not positive definite');
    if (info < 0) throw new Error('cuSOLVER potrf: invalid argument ' + (-info));
    const L = new Float32Array(n * n);
    copyDeviceToHost(L, dA);
    return L;
  } finally { s.free(); }
}

export function csGetrf(Acm: ArrayBufferView, n: number): { LU: Float32Array; ipiv: Int32Array } {
  const h = handle();
  const s = scope();
  try {
    const dA = s.up(Acm);
    const lwork = [0];
    ck('getrf_bufferSize', c.getrfBufferSize(h, n, n, dA, n, lwork));
    const dWork = s.out(lwork[0] * 4), dIpiv = s.out(n * 4), dInfo = s.out(4);
    ck('getrf', c.getrf(h, n, n, dA, n, dWork, dIpiv, dInfo));
    sync();
    const LU = new Float32Array(n * n), ipiv = new Int32Array(n);
    copyDeviceToHost(LU, dA); copyDeviceToHost(ipiv, dIpiv);
    return { LU, ipiv };
  } finally { s.free(); }
}

export function csSolveLU(Acm: ArrayBufferView, n: number, Bcm: ArrayBufferView, nrhs: number, trans: number): Float32Array {
  const h = handle();
  const s = scope();
  try {
    const dA = s.up(Acm);
    const lwork = [0];
    ck('getrf_bufferSize', c.getrfBufferSize(h, n, n, dA, n, lwork));
    const dWork = s.out(lwork[0] * 4), dIpiv = s.out(n * 4), dInfo = s.out(4);
    ck('getrf', c.getrf(h, n, n, dA, n, dWork, dIpiv, dInfo));
    sync();
    let info = readInt(dInfo);
    if (info > 0) throw new Error('linalg: matrix is singular');
    const dB = s.up(Bcm);
    ck('getrs', c.getrs(h, trans, n, nrhs, dA, n, dIpiv, dB, n, dInfo));
    sync();
    info = readInt(dInfo);
    if (info < 0) throw new Error('cuSOLVER getrs: invalid argument ' + (-info));
    const X = new Float32Array(n * nrhs);
    copyDeviceToHost(X, dB);
    return X;
  } finally { s.free(); }
}
