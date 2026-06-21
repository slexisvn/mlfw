import koffi from 'koffi';
import { readdirSync, existsSync } from 'fs';
import { join } from 'path';
import { getDevice } from './device.js';

function resolveLib(pattern, fallback) {
  const roots = [];
  if (process.env.CUDA_PATH) roots.push(process.env.CUDA_PATH);
  const toolkit = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA';
  if (existsSync(toolkit)) for (const v of readdirSync(toolkit)) roots.push(join(toolkit, v));
  for (const root of roots) {
    const bin = join(root, 'bin');
    if (!existsSync(bin)) continue;
    const matches = readdirSync(bin).filter(f => pattern.test(f));
    if (matches.length > 0) {
      if (!process.env.PATH.includes(bin)) process.env.PATH = bin + ';' + process.env.PATH;
      return matches.sort().pop();
    }
  }
  return fallback;
}

const rt = koffi.load(resolveLib(/^cudart64_\d+\.dll$/, 'cudart64_12.dll'));
const blas = koffi.load(resolveLib(/^cublas64_\d+\.dll$/, 'cublas64_12.dll'));

const cudaSetDevice = rt.func('int cudaSetDevice(int d)');
const cudaMalloc = rt.func('int cudaMalloc(_Out_ void **p, size_t s)');
const cudaMemcpy = rt.func('int cudaMemcpy(void *dst, void *src, size_t n, int kind)');
const cudaFree = rt.func('int cudaFree(void *p)');
const cudaDeviceSynchronize = rt.func('int cudaDeviceSynchronize()');
const cublasCreate = blas.func('int cublasCreate_v2(_Out_ void **h)');
const cublasSetStream = blas.func('int cublasSetStream_v2(void *h, void *streamId)');
const cublasSetMathMode = blas.func('int cublasSetMathMode(void *h, int mode)');
const cublasSgemm = blas.func('int cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, float *alpha, void *A, int lda, void *B, int ldb, float *beta, void *C, int ldc)');
const cublasSgemmU = blas.func('int cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, float *alpha, uint64 A, int lda, uint64 B, int ldb, float *beta, uint64 C, int ldc)');
const cublasSgemmBatchedU = blas.func('int cublasSgemmStridedBatched(void *h, int ta, int tb, int m, int n, int k, float *alpha, uint64 A, int lda, int64 sa, uint64 B, int ldb, int64 sb, float *beta, uint64 C, int ldc, int64 sc, int batch)');

const H2D = 1, D2H = 2;
const OP_N = 0, OP_T = 1;
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
    cublasSetMathMode(_handle, 3);
  }
  return _handle;
}

const _pool = new Map();
function acquire(bytes) {
  const freeList = _pool.get(bytes);
  if (freeList && freeList.length > 0) return freeList.pop();
  const p = [null];
  if (cudaMalloc(p, bytes)) throw new Error('cudaMalloc failed');
  return p[0];
}
function release(ptr, bytes) {
  let freeList = _pool.get(bytes);
  if (!freeList) { freeList = []; _pool.set(bytes, freeList); }
  freeList.push(ptr);
}

export function cublasMatmulDevice(M, N, K, dA, dB, dC, transB = false) {
  cudaSetDevice(0);
  const h = handle();
  const opB = transB ? 1 : 0;
  const ldB = transB ? K : N;
  const status = cublasSgemm(h, opB, 0, N, M, K, ALPHA, dB, ldB, dA, K, BETA, dC, N);
  if (status !== 0) throw new Error('cublasSgemm failed: ' + status);
}

export function cublasGemmDevice(M, N, K, dA, transA, dB, transB, dC) {
  cudaSetDevice(0);
  const h = handle();
  const opA = transA ? OP_T : OP_N, opB = transB ? OP_T : OP_N;
  const lda = transA ? M : K, ldb = transB ? K : N;
  const status = cublasSgemmU(h, opB, opA, N, M, K, ALPHA, dB, ldb, dA, lda, BETA, dC, N);
  if (status !== 0) throw new Error('cublasSgemm failed: ' + status);
}

export function cublasGemmBatchedDevice(batch, M, N, K, dA, sA, transA, dB, sB, transB, dC, sC) {
  cudaSetDevice(0);
  const h = handle();
  const opA = transA ? OP_T : OP_N, opB = transB ? OP_T : OP_N;
  const lda = transA ? M : K, ldb = transB ? K : N;
  const status = cublasSgemmBatchedU(h, opB, opA, N, M, K, ALPHA, dB, ldb, BigInt(sB), dA, lda, BigInt(sA), BETA, dC, N, BigInt(sC), batch);
  if (status !== 0) throw new Error('cublasSgemmStridedBatched failed: ' + status);
}

export function cublasMatmul(M, N, K, A, B, C, transB = false) {
  cudaSetDevice(0);
  const h = handle();
  const ad = acquire(A.byteLength), bd = acquire(B.byteLength), cd = acquire(C.byteLength);
  try {
    cudaMemcpy(ad, A, A.byteLength, H2D);
    cudaMemcpy(bd, B, B.byteLength, H2D);
    const opB = transB ? 1 : 0;
    const ldB = transB ? K : N;
    const status = cublasSgemm(h, opB, 0, N, M, K, ALPHA, bd, ldB, ad, K, BETA, cd, N);
    if (status !== 0) throw new Error('cublasSgemm failed: ' + status);
    cudaDeviceSynchronize();
    cudaMemcpy(C, cd, C.byteLength, D2H);
  } finally {
    release(ad, A.byteLength); release(bd, B.byteLength); release(cd, C.byteLength);
  }
}
