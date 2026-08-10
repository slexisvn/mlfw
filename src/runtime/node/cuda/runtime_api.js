import koffi from 'koffi';
import { loadCudaLib, CUDART_SPEC } from './lib_resolver.js';
import { copyHostToDevice, copyDeviceToHost } from './memory.js';
import { cu, checkCU } from './ffi.js';

const rt = koffi.load(loadCudaLib(CUDART_SPEC));

const cudaSetDevice = rt.func('int cudaSetDevice(int d)');

let _h2dObserver = null;
export function setH2DObserver(fn) { _h2dObserver = fn; }

export function setDevice() {
  cudaSetDevice(0);
}

export function devSync() {
  checkCU('cuCtxSynchronize', cu.ctxSynchronize());
}

export function devH2D(dptr, hostView) {
  copyHostToDevice(dptr, hostView);
  if (_h2dObserver) _h2dObserver(hostView.byteLength);
}

export function devD2H(hostView, dptr) {
  copyDeviceToHost(hostView, dptr);
}
