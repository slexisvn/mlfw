import { cu, checkCU } from './ffi.js';
import type { DevicePtr } from './ffi.js';
import { getDevice } from './device.js';

const CUDA_ERROR_OUT_OF_MEMORY = 2;

export function alloc(bytes: number): DevicePtr {
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

export function copyHostToDevice(dptr: DevicePtr, hostView: ArrayBufferView): void {
  checkCU('cuMemcpyHtoD', cu.memcpyHtoD(dptr, hostView, hostView.byteLength));
}

export function copyDeviceToHost(hostView: ArrayBufferView, dptr: DevicePtr): void {
  checkCU('cuMemcpyDtoH', cu.memcpyDtoH(hostView, dptr, hostView.byteLength));
}

export function copyHostToDeviceAsync(dptr: DevicePtr, hostView: ArrayBufferView): void {
  checkCU('cuMemcpyHtoDAsync', cu.memcpyHtoDAsync(dptr, hostView, hostView.byteLength, getDevice().stream));
}

export function copyDeviceToHostAsync(hostView: ArrayBufferView, dptr: DevicePtr): void {
  checkCU('cuMemcpyDtoHAsync', cu.memcpyDtoHAsync(hostView, dptr, hostView.byteLength, getDevice().stream));
}

export function free(dptr: DevicePtr): void {
  checkCU('cuMemFree', cu.memFree(dptr));
}

const POOL_FRACTION_OF_FREE = 0.10;
const MIN_POOL_BYTES = 64 * 1024 * 1024;

const _pool = new Map<number, DevicePtr[]>();
let _pooledBytes = 0;
let _liveBytes = 0;
let _peakLiveBytes = 0;
let _poolCap = 0;

export function pooledBytes(): number { return _pooledBytes; }
export function liveBytes(): number { return _liveBytes; }
export function peakLiveBytes(): number { return _peakLiveBytes; }
export function resetPeakLiveBytes(): void { _peakLiveBytes = _liveBytes; }
export function poolBuckets(): number { return _pool.size; }
export function setPoolLimit(bytes: number): void { _poolCap = Math.max(bytes, 0); _trimPool(); }

function poolLimit(): number {
  if (!_poolCap) {
    getDevice();
    const freeMem = [0n], total = [0n];
    checkCU('cuMemGetInfo', cu.memGetInfo(freeMem, total));
    _poolCap = Math.max(Math.floor(Number(freeMem[0]) * POOL_FRACTION_OF_FREE), MIN_POOL_BYTES);
  }
  return _poolCap;
}

function sizeClass(bytes: number): number {
  if (bytes <= 512) return 512;
  const step = 2 ** Math.max(9, Math.ceil(Math.log2(bytes)) - 3);
  return Math.ceil(bytes / step) * step;
}

function _touch(cls: number, freeList: DevicePtr[]): void {
  _pool.delete(cls);
  _pool.set(cls, freeList);
}

function _trimPool(): void {
  const limit = poolLimit();
  if (_pooledBytes <= limit) return;
  for (const [cls, freeList] of _pool) {
    while (freeList.length > 0 && _pooledBytes > limit) {
      free(freeList.pop()!);
      _pooledBytes -= cls;
    }
    if (freeList.length === 0) _pool.delete(cls);
    if (_pooledBytes <= limit) return;
  }
}

export function acquire(bytes: number): DevicePtr {
  const cls = sizeClass(bytes);
  const freeList = _pool.get(cls);
  let dptr: DevicePtr;
  if (freeList && freeList.length > 0) {
    _pooledBytes -= cls;
    _touch(cls, freeList);
    dptr = freeList.pop()!;
  } else {
    dptr = alloc(cls);
  }
  _liveBytes += cls;
  if (_liveBytes > _peakLiveBytes) _peakLiveBytes = _liveBytes;
  return dptr;
}

export function release(dptr: DevicePtr, bytes: number): void {
  const cls = sizeClass(bytes);
  let freeList = _pool.get(cls);
  if (!freeList) freeList = [];
  freeList.push(dptr);
  _touch(cls, freeList);
  _pooledBytes += cls;
  _liveBytes -= cls;
  _trimPool();
}

export function drainPool(): number {
  let freed = 0;
  for (const freeList of _pool.values()) {
    for (const dptr of freeList) { free(dptr); freed++; }
  }
  _pool.clear();
  _pooledBytes = 0;
  return freed;
}
