import { cu, checkCU } from './ffi.js';
import { getDevice } from './device.js';

const CUDA_ERROR_OUT_OF_MEMORY = 2;

export function alloc(bytes) {
  const dptr = [0n];
  let code = cu.memAlloc(dptr, bytes);
  if (code === CUDA_ERROR_OUT_OF_MEMORY) {
    drainPool();
    code = cu.memAlloc(dptr, bytes);
  }
  if (code !== 0) {
    const free = [0n], total = [0n];
    cu.memGetInfo(free, total);
    throw new Error(`cuMemAlloc failed for ${bytes} bytes (driver error ${code}); device has ${Number(free[0])} of ${Number(total[0])} bytes free`);
  }
  return dptr[0];
}

export function copyHostToDevice(dptr, hostView) {
  checkCU('cuMemcpyHtoD', cu.memcpyHtoD(dptr, hostView, hostView.byteLength));
}

export function copyDeviceToHost(hostView, dptr) {
  checkCU('cuMemcpyDtoH', cu.memcpyDtoH(hostView, dptr, hostView.byteLength));
}

export function copyHostToDeviceAsync(dptr, hostView) {
  checkCU('cuMemcpyHtoDAsync', cu.memcpyHtoDAsync(dptr, hostView, hostView.byteLength, getDevice().stream));
}

export function copyDeviceToHostAsync(hostView, dptr) {
  checkCU('cuMemcpyDtoHAsync', cu.memcpyDtoHAsync(hostView, dptr, hostView.byteLength, getDevice().stream));
}

export function free(dptr) {
  checkCU('cuMemFree', cu.memFree(dptr));
}

const POOL_FRACTION_OF_FREE = 0.10;
const MIN_POOL_BYTES = 64 * 1024 * 1024;

const _pool = new Map();
let _pooledBytes = 0;
let _liveBytes = 0;
let _poolCap = 0;

export function pooledBytes() { return _pooledBytes; }
export function liveBytes() { return _liveBytes; }
export function poolBuckets() { return _pool.size; }
export function setPoolLimit(bytes) { _poolCap = Math.max(bytes, 0); _trimPool(); }

function poolLimit() {
  if (!_poolCap) {
    getDevice();
    const freeMem = [0n], total = [0n];
    checkCU('cuMemGetInfo', cu.memGetInfo(freeMem, total));
    _poolCap = Math.max(Math.floor(Number(freeMem[0]) * POOL_FRACTION_OF_FREE), MIN_POOL_BYTES);
  }
  return _poolCap;
}

function sizeClass(bytes) {
  if (bytes <= 512) return 512;
  const step = 2 ** Math.max(9, Math.ceil(Math.log2(bytes)) - 3);
  return Math.ceil(bytes / step) * step;
}

function _touch(cls, freeList) {
  _pool.delete(cls);
  _pool.set(cls, freeList);
}

function _trimPool() {
  const limit = poolLimit();
  if (_pooledBytes <= limit) return;
  for (const [cls, freeList] of _pool) {
    while (freeList.length > 0 && _pooledBytes > limit) {
      free(freeList.pop());
      _pooledBytes -= cls;
    }
    if (freeList.length === 0) _pool.delete(cls);
    if (_pooledBytes <= limit) return;
  }
}

export function acquire(bytes) {
  const cls = sizeClass(bytes);
  const freeList = _pool.get(cls);
  if (freeList && freeList.length > 0) {
    _pooledBytes -= cls;
    _liveBytes += cls;
    _touch(cls, freeList);
    return freeList.pop();
  }
  const dptr = alloc(cls);
  _liveBytes += cls;
  return dptr;
}

export function release(dptr, bytes) {
  const cls = sizeClass(bytes);
  let freeList = _pool.get(cls);
  if (!freeList) freeList = [];
  freeList.push(dptr);
  _touch(cls, freeList);
  _pooledBytes += cls;
  _liveBytes -= cls;
  _trimPool();
}

export function drainPool() {
  let freed = 0;
  for (const freeList of _pool.values()) {
    for (const dptr of freeList) { free(dptr); freed++; }
  }
  _pool.clear();
  _pooledBytes = 0;
  return freed;
}
