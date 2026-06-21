import koffi from 'koffi';
import { readdirSync, existsSync } from 'fs';
import { join } from 'path';

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

const cudaSetDevice = rt.func('int cudaSetDevice(int d)');
const cudaMalloc = rt.func('int cudaMalloc(_Out_ void **p, size_t s)');
const cudaMemcpy = rt.func('int cudaMemcpy(void *dst, void *src, size_t n, int kind)');
const cudaFree = rt.func('int cudaFree(void *p)');
const cudaDeviceSynchronize = rt.func('int cudaDeviceSynchronize()');
const cudaStreamCreate = rt.func('int cudaStreamCreate(_Out_ void **s)');
const cudaStreamBeginCapture = rt.func('int cudaStreamBeginCapture(void *s, int mode)');
const cudaStreamEndCapture = rt.func('int cudaStreamEndCapture(void *s, _Out_ void **g)');
const cudaGraphInstantiate = rt.func('int cudaGraphInstantiate(_Out_ void **e, void *g, uint64 flags)');
const cudaGraphLaunch = rt.func('int cudaGraphLaunch(void *e, void *s)');
const cudaStreamSynchronize = rt.func('int cudaStreamSynchronize(void *s)');

const H2D = 1, D2H = 2;

export function setDevice() {
  cudaSetDevice(0);
}

let _captureStream = null;
export function getCaptureStream() {
  if (!_captureStream) { const s = [null]; if (cudaStreamCreate(s) !== 0) throw new Error('cudaStreamCreate failed'); _captureStream = s[0]; }
  return _captureStream;
}
export function beginCapture(stream) {
  const r = cudaStreamBeginCapture(stream, 0);
  if (r !== 0) throw new Error('cudaStreamBeginCapture failed: ' + r);
}
export function endCaptureInstantiate(stream) {
  const g = [null];
  let r = cudaStreamEndCapture(stream, g);
  if (r !== 0) throw new Error('cudaStreamEndCapture failed: ' + r);
  const e = [null];
  r = cudaGraphInstantiate(e, g[0], 0n);
  if (r !== 0) throw new Error('cudaGraphInstantiate failed: ' + r);
  return e[0];
}
export function graphLaunch(exec, stream) {
  const r = cudaGraphLaunch(exec, stream);
  if (r !== 0) throw new Error('cudaGraphLaunch failed: ' + r);
}
export function streamSync(stream) {
  const r = cudaStreamSynchronize(stream);
  if (r !== 0) throw new Error('cudaStreamSynchronize failed: ' + r);
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
