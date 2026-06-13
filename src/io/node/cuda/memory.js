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
