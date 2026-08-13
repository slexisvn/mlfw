import koffi from 'koffi';
import { getDevice } from './device.js';
import type { CudaHandle, DevicePtr } from './ffi.js';
import { setDevice, devSync } from './runtime_api.js';
import { acquire, release, copyHostToDevice, copyDeviceToHost } from './memory.js';
import { loadCudaLib, CUBLAS_SPEC } from './lib_resolver.js';

const blas = koffi.load(loadCudaLib(CUBLAS_SPEC));

const cublasCreate: (h: (CudaHandle | null)[]) => number = blas.func('int cublasCreate_v2(_Out_ void **h)');
const cublasDestroy: (h: CudaHandle | null) => number = blas.func('int cublasDestroy_v2(void *h)');
const cublasSetStream: (h: CudaHandle | null, streamId: CudaHandle | null) => number = blas.func('int cublasSetStream_v2(void *h, void *streamId)');
const cublasSetMathMode: (h: CudaHandle | null, mode: number) => number = blas.func('int cublasSetMathMode(void *h, int mode)');
const cublasSgemm: (h: CudaHandle | null, ta: number, tb: number, m: number, n: number, k: number, alpha: Float32Array, A: DevicePtr, lda: number, B: DevicePtr, ldb: number, beta: Float32Array, C: DevicePtr, ldc: number) => number = blas.func('int cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, float *alpha, uint64 A, int lda, uint64 B, int ldb, float *beta, uint64 C, int ldc)');
const cublasSgemmStridedBatched: (h: CudaHandle | null, ta: number, tb: number, m: number, n: number, k: number, alpha: Float32Array, A: DevicePtr, lda: number, sa: bigint, B: DevicePtr, ldb: number, sb: bigint, beta: Float32Array, C: DevicePtr, ldc: number, sc: bigint, batch: number) => number = blas.func('int cublasSgemmStridedBatched(void *h, int ta, int tb, int m, int n, int k, float *alpha, uint64 A, int lda, int64 sa, uint64 B, int ldb, int64 sb, float *beta, uint64 C, int ldc, int64 sc, int batch)');

const OP_N = 0, OP_T = 1;
const TENSOR_OP_MATH = 3;
const ALPHA = new Float32Array([1]);
const BETA = new Float32Array([0]);

let _handle: CudaHandle | null = null;
function handle(): CudaHandle | null {
  if (!_handle) {
    const h: (CudaHandle | null)[] = [null];
    const status = cublasCreate(h);
    if (status !== 0) throw new Error('cublasCreate failed: ' + status);
    _handle = h[0];
    cublasSetStream(_handle, getDevice().stream);
    cublasSetMathMode(_handle, TENSOR_OP_MATH);
  }
  return _handle;
}

export function destroyCublas(): void {
  if (!_handle) return;
  cublasDestroy(_handle);
  _handle = null;
}

function gemm(M: number, N: number, K: number, dA: DevicePtr, transA: boolean, dB: DevicePtr, transB: boolean, dC: DevicePtr): void {
  const opA = transA ? OP_T : OP_N, opB = transB ? OP_T : OP_N;
  const lda = transA ? M : K, ldb = transB ? K : N;
  const status = cublasSgemm(handle(), opB, opA, N, M, K, ALPHA, dB, ldb, dA, lda, BETA, dC, N);
  if (status !== 0) throw new Error('cublasSgemm failed: ' + status);
}

export function cublasMatmulDevice(M: number, N: number, K: number, dA: DevicePtr, dB: DevicePtr, dC: DevicePtr, transB = false): void {
  setDevice();
  gemm(M, N, K, dA, false, dB, transB, dC);
}

export function cublasGemmDevice(M: number, N: number, K: number, dA: DevicePtr, transA: boolean, dB: DevicePtr, transB: boolean, dC: DevicePtr): void {
  setDevice();
  gemm(M, N, K, dA, transA, dB, transB, dC);
}

export function cublasGemmBatchedDevice(batch: number, M: number, N: number, K: number, dA: DevicePtr, sA: number, transA: boolean, dB: DevicePtr, sB: number, transB: boolean, dC: DevicePtr, sC: number): void {
  setDevice();
  const opA = transA ? OP_T : OP_N, opB = transB ? OP_T : OP_N;
  const lda = transA ? M : K, ldb = transB ? K : N;
  const status = cublasSgemmStridedBatched(handle(), opB, opA, N, M, K, ALPHA, dB, ldb, BigInt(sB), dA, lda, BigInt(sA), BETA, dC, N, BigInt(sC), batch);
  if (status !== 0) throw new Error('cublasSgemmStridedBatched failed: ' + status);
}

export function cublasMatmul(M: number, N: number, K: number, A: ArrayBufferView, B: ArrayBufferView, C: ArrayBufferView, transB = false): void {
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
