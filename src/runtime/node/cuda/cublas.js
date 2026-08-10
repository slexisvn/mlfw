import koffi from 'koffi';
import { getDevice } from './device.js';
import { setDevice, devSync } from './runtime_api.js';
import { acquire, release, copyHostToDevice, copyDeviceToHost } from './memory.js';
import { loadCudaLib, CUBLAS_SPEC } from './lib_resolver.js';

const blas = koffi.load(loadCudaLib(CUBLAS_SPEC));

const cublasCreate = blas.func('int cublasCreate_v2(_Out_ void **h)');
const cublasDestroy = blas.func('int cublasDestroy_v2(void *h)');
const cublasSetStream = blas.func('int cublasSetStream_v2(void *h, void *streamId)');
const cublasSetMathMode = blas.func('int cublasSetMathMode(void *h, int mode)');
const cublasSgemm = blas.func('int cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, float *alpha, uint64 A, int lda, uint64 B, int ldb, float *beta, uint64 C, int ldc)');
const cublasSgemmStridedBatched = blas.func('int cublasSgemmStridedBatched(void *h, int ta, int tb, int m, int n, int k, float *alpha, uint64 A, int lda, int64 sa, uint64 B, int ldb, int64 sb, float *beta, uint64 C, int ldc, int64 sc, int batch)');

const OP_N = 0, OP_T = 1;
const TENSOR_OP_MATH = 3;
const ALPHA = new Float32Array([1]);
const BETA = new Float32Array([0]);

let _handle = null;
function handle() {
  if (!_handle) {
    const h = [null];
    const status = cublasCreate(h);
    if (status !== 0) throw new Error('cublasCreate failed: ' + status);
    _handle = h[0];
    cublasSetStream(_handle, getDevice().stream);
    cublasSetMathMode(_handle, TENSOR_OP_MATH);
  }
  return _handle;
}

export function destroyCublas() {
  if (!_handle) return;
  cublasDestroy(_handle);
  _handle = null;
}

function gemm(M, N, K, dA, transA, dB, transB, dC) {
  const opA = transA ? OP_T : OP_N, opB = transB ? OP_T : OP_N;
  const lda = transA ? M : K, ldb = transB ? K : N;
  const status = cublasSgemm(handle(), opB, opA, N, M, K, ALPHA, dB, ldb, dA, lda, BETA, dC, N);
  if (status !== 0) throw new Error('cublasSgemm failed: ' + status);
}

export function cublasMatmulDevice(M, N, K, dA, dB, dC, transB = false) {
  setDevice();
  gemm(M, N, K, dA, false, dB, transB, dC);
}

export function cublasGemmDevice(M, N, K, dA, transA, dB, transB, dC) {
  setDevice();
  gemm(M, N, K, dA, transA, dB, transB, dC);
}

export function cublasGemmBatchedDevice(batch, M, N, K, dA, sA, transA, dB, sB, transB, dC, sC) {
  setDevice();
  const opA = transA ? OP_T : OP_N, opB = transB ? OP_T : OP_N;
  const lda = transA ? M : K, ldb = transB ? K : N;
  const status = cublasSgemmStridedBatched(handle(), opB, opA, N, M, K, ALPHA, dB, ldb, BigInt(sB), dA, lda, BigInt(sA), BETA, dC, N, BigInt(sC), batch);
  if (status !== 0) throw new Error('cublasSgemmStridedBatched failed: ' + status);
}

export function cublasMatmul(M, N, K, A, B, C, transB = false) {
  setDevice();
  const ad = acquire(A.byteLength), bd = acquire(B.byteLength), cd = acquire(C.byteLength);
  try {
    copyHostToDevice(ad, A);
    copyHostToDevice(bd, B);
    gemm(M, N, K, ad, false, bd, transB, cd);
    devSync();
    copyDeviceToHost(C, cd);
  } finally {
    release(ad, A.byteLength);
    release(bd, B.byteLength);
    release(cd, C.byteLength);
  }
}
