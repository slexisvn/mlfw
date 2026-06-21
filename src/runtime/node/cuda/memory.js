import { cu, checkCU } from './ffi.js';

export function alloc(bytes) {
  const dptr = [0n];
  checkCU('cuMemAlloc', cu.memAlloc(dptr, bytes));
  return dptr[0];
}

export function copyHostToDevice(dptr, hostView) {
  checkCU('cuMemcpyHtoD', cu.memcpyHtoD(dptr, hostView, hostView.byteLength));
}

export function copyDeviceToHost(hostView, dptr) {
  checkCU('cuMemcpyDtoH', cu.memcpyDtoH(hostView, dptr, hostView.byteLength));
}

export function free(dptr) {
  cu.memFree(dptr);
}

const _pool = new Map();

export function acquire(bytes) {
  const freeList = _pool.get(bytes);
  if (freeList && freeList.length > 0) return freeList.pop();
  return alloc(bytes);
}

export function release(dptr, bytes) {
  let freeList = _pool.get(bytes);
  if (!freeList) { freeList = []; _pool.set(bytes, freeList); }
  freeList.push(dptr);
}
