import koffi from 'koffi';
import { loadCudaLib, CUDART_SPEC } from './lib_resolver.js';
import { copyHostToDevice, copyDeviceToHost } from './memory.js';
import { cu, checkCU } from './ffi.js';
import type { DevicePtr } from './ffi.js';

const rt = koffi.load(loadCudaLib(CUDART_SPEC));

const cudaSetDevice = rt.func('int cudaSetDevice(int d)');

type H2DObserver = (bytes: number) => void;

let _h2dObserver: H2DObserver | null = null;
export function setH2DObserver(fn: H2DObserver | null): void { _h2dObserver = fn; }

export function setDevice(): void {
  cudaSetDevice(0);
}

export function devSync(): void {
  checkCU('cuCtxSynchronize', cu.ctxSynchronize());
}

export function devH2D(dptr: DevicePtr, hostView: ArrayBufferView): void {
  copyHostToDevice(dptr, hostView);
  if (_h2dObserver) _h2dObserver(hostView.byteLength);
}

export function devD2H(hostView: ArrayBufferView, dptr: DevicePtr): void {
  copyDeviceToHost(hostView, dptr);
}
