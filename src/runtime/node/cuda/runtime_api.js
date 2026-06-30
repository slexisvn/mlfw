import koffi from 'koffi';
import { loadCudaLib, CUDART_SPEC } from './lib_resolver.js';

const rt = koffi.load(loadCudaLib(CUDART_SPEC));

const cudaSetDevice = rt.func('int cudaSetDevice(int d)');
const cudaMalloc = rt.func('int cudaMalloc(_Out_ void **p, size_t s)');
const cudaMemcpy = rt.func('int cudaMemcpy(void *dst, void *src, size_t n, int kind)');
const cudaFree = rt.func('int cudaFree(void *p)');
const cudaDeviceSynchronize = rt.func('int cudaDeviceSynchronize()');

const H2D = 1, D2H = 2;

let _h2dObserver = null;
export function setH2DObserver(fn) { _h2dObserver = fn; }

export function setDevice() {
  cudaSetDevice(0);
}

export function devSync() {
  const status = cudaDeviceSynchronize();
  if (status !== 0) throw new Error('cudaDeviceSynchronize failed: ' + status);
}

export function devAlloc(bytes) {
  const p = [null];
  const status = cudaMalloc(p, bytes);
  if (status !== 0) throw new Error('cudaMalloc failed: ' + status);
  return p[0];
}

export function devFree(ptr) {
  cudaFree(ptr);
}

export function devH2D(dptr, hostView) {
  const status = cudaMemcpy(dptr, hostView, hostView.byteLength, H2D);
  if (status !== 0) throw new Error('cudaMemcpy H2D failed: ' + status);
  if (_h2dObserver) _h2dObserver(hostView.byteLength);
}

export function devD2H(hostView, dptr) {
  const status = cudaMemcpy(hostView, dptr, hostView.byteLength, D2H);
  if (status !== 0) throw new Error('cudaMemcpy D2H failed: ' + status);
}

export function devAddr(ptr) {
  if (typeof ptr === 'bigint') return ptr;
  if (typeof ptr === 'number') return BigInt(ptr);
  return koffi.address(ptr);
}

const _pool = new Map();

export function acquireDevice(bytes) {
  const freeList = _pool.get(bytes);
  if (freeList && freeList.length > 0) return freeList.pop();
  return devAlloc(bytes);
}

export function releaseDevice(ptr, bytes) {
  let freeList = _pool.get(bytes);
  if (!freeList) { freeList = []; _pool.set(bytes, freeList); }
  freeList.push(ptr);
}
