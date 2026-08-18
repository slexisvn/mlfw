import koffi from 'koffi';
import { loadCudaLib, cudaIncludeDir, DRIVER_SPEC, NVRTC_SPEC } from './lib_resolver.js';

export { cudaIncludeDir };

import type { CudaHandle, DevicePtr } from '../../io.js';

export type { CudaHandle, DevicePtr };

export interface CuApi {
  init(flags: number): number;
  deviceGet(dev: number[], ordinal: number): number;
  deviceGetAttribute(pi: number[], attrib: number, dev: number): number;
  primaryCtxRetain(pctx: (CudaHandle | null)[], dev: number): number;
  primaryCtxRelease(dev: number): number;
  ctxSynchronize(): number;
  moduleLoadData(mod: (CudaHandle | null)[], image: Uint8Array): number;
  moduleGetFunction(func: (CudaHandle | null)[], mod: CudaHandle | null, name: string): number;
  memGetInfo(free: bigint[], total: bigint[]): number;
  memAlloc(dptr: bigint[], bytes: number): number;
  memFree(dptr: DevicePtr): number;
  memcpyHtoD(dst: DevicePtr, src: ArrayBufferView, n: number): number;
  memcpyDtoH(dst: ArrayBufferView, src: DevicePtr, n: number): number;
  memcpyDtoD(dst: DevicePtr, src: DevicePtr, n: number): number;
  memcpyHtoDAsync(dst: DevicePtr, src: ArrayBufferView, n: number, stream: CudaHandle | null): number;
  memcpyDtoHAsync(dst: ArrayBufferView, src: DevicePtr, n: number, stream: CudaHandle | null): number;
  memcpyDtoDAsync(dst: DevicePtr, src: DevicePtr, n: number, stream: CudaHandle | null): number;
  memsetD8(dst: DevicePtr, uc: number, n: number): number;
  memsetD8Async(dst: DevicePtr, uc: number, n: number, stream: CudaHandle | null): number;
  launchKernel(f: CudaHandle | null, gx: number, gy: number, gz: number, bx: number, by: number, bz: number, shmem: number, stream: CudaHandle | null, params: Uint8Array[], extra: null): number;
  streamCreate(stream: (CudaHandle | null)[], flags: number): number;
  streamSynchronize(stream: CudaHandle | null): number;
  ctxSetCurrent(ctx: CudaHandle | null): number;
  streamBeginCapture(stream: CudaHandle | null, mode: number): number;
  streamEndCapture(stream: CudaHandle | null, graph: (CudaHandle | null)[]): number;
  graphInstantiate(exec: (CudaHandle | null)[], graph: CudaHandle | null, flags: bigint): number;
  graphLaunch(exec: CudaHandle | null, stream: CudaHandle | null): number;
  graphExecDestroy(exec: CudaHandle | null): number;
  graphDestroy(graph: CudaHandle | null): number;
}

export interface NvApi {
  createProgram(prog: (CudaHandle | null)[], src: string, name: string, n: number, h: null, inc: null): number;
  compileProgram(prog: CudaHandle | null, n: number, opts: string[]): number;
  destroyProgram(prog: (CudaHandle | null)[]): number;
  getPTXSize(prog: CudaHandle | null, sz: bigint[]): number;
  getPTX(prog: CudaHandle | null, ptx: Uint8Array): number;
  getProgramLogSize(prog: CudaHandle | null, sz: bigint[]): number;
  getProgramLog(prog: CudaHandle | null, log: Uint8Array): number;
  version(major: number[], minor: number[]): number;
}

const drv = koffi.load(loadCudaLib(DRIVER_SPEC));
const nvrtc = koffi.load(loadCudaLib(NVRTC_SPEC));

export const cu: CuApi = {
  init: drv.func('int cuInit(uint)'),
  deviceGet: drv.func('int cuDeviceGet(_Out_ int *dev, int ordinal)'),
  deviceGetAttribute: drv.func('int cuDeviceGetAttribute(_Out_ int *pi, int attrib, int dev)'),
  primaryCtxRetain: drv.func('int cuDevicePrimaryCtxRetain(_Out_ void **pctx, int dev)'),
  primaryCtxRelease: drv.func('int cuDevicePrimaryCtxRelease_v2(int dev)'),
  ctxSynchronize: drv.func('int cuCtxSynchronize()'),
  moduleLoadData: drv.func('int cuModuleLoadData(_Out_ void **mod, void *image)'),
  moduleGetFunction: drv.func('int cuModuleGetFunction(_Out_ void **func, void *mod, str name)'),
  memGetInfo: drv.func('int cuMemGetInfo_v2(_Out_ uint64 *free, _Out_ uint64 *total)'),
  memAlloc: drv.func('int cuMemAlloc_v2(_Out_ uint64 *dptr, size_t bytes)'),
  memFree: drv.func('int cuMemFree_v2(uint64 dptr)'),
  memcpyHtoD: drv.func('int cuMemcpyHtoD_v2(uint64 dst, void *src, size_t n)'),
  memcpyDtoH: drv.func('int cuMemcpyDtoH_v2(void *dst, uint64 src, size_t n)'),
  memcpyDtoD: drv.func('int cuMemcpyDtoD_v2(uint64 dst, uint64 src, size_t n)'),
  memcpyHtoDAsync: drv.func('int cuMemcpyHtoDAsync_v2(uint64 dst, void *src, size_t n, void *stream)'),
  memcpyDtoHAsync: drv.func('int cuMemcpyDtoHAsync_v2(void *dst, uint64 src, size_t n, void *stream)'),
  memcpyDtoDAsync: drv.func('int cuMemcpyDtoDAsync_v2(uint64 dst, uint64 src, size_t n, void *stream)'),
  memsetD8: drv.func('int cuMemsetD8_v2(uint64 dst, uint8 uc, size_t n)'),
  memsetD8Async: drv.func('int cuMemsetD8Async(uint64 dst, uint8 uc, size_t n, void *stream)'),
  launchKernel: drv.func('int cuLaunchKernel(void *f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shmem, void *stream, void **params, void **extra)'),
  streamCreate: drv.func('int cuStreamCreate(_Out_ void **stream, uint flags)'),
  streamSynchronize: drv.func('int cuStreamSynchronize(void *stream)'),
  ctxSetCurrent: drv.func('int cuCtxSetCurrent(void *ctx)'),
  streamBeginCapture: drv.func('int cuStreamBeginCapture_v2(void *stream, int mode)'),
  streamEndCapture: drv.func('int cuStreamEndCapture(void *stream, _Out_ void **graph)'),
  graphInstantiate: drv.func('int cuGraphInstantiateWithFlags(_Out_ void **exec, void *graph, uint64 flags)'),
  graphLaunch: drv.func('int cuGraphLaunch(void *exec, void *stream)'),
  graphExecDestroy: drv.func('int cuGraphExecDestroy(void *exec)'),
  graphDestroy: drv.func('int cuGraphDestroy(void *graph)'),
};

export const nv: NvApi = {
  createProgram: nvrtc.func('int nvrtcCreateProgram(_Out_ void **prog, str src, str name, int n, void *h, void *inc)'),
  compileProgram: nvrtc.func('int nvrtcCompileProgram(void *prog, int n, str *opts)'),
  destroyProgram: nvrtc.func('int nvrtcDestroyProgram(void **prog)'),
  getPTXSize: nvrtc.func('int nvrtcGetPTXSize(void *prog, _Out_ size_t *sz)'),
  getPTX: nvrtc.func('int nvrtcGetPTX(void *prog, _Out_ uint8_t *ptx)'),
  getProgramLogSize: nvrtc.func('int nvrtcGetProgramLogSize(void *prog, _Out_ size_t *sz)'),
  getProgramLog: nvrtc.func('int nvrtcGetProgramLog(void *prog, _Out_ uint8_t *log)'),
  version: nvrtc.func('int nvrtcVersion(_Out_ int *major, _Out_ int *minor)'),
};

export const ATTR_CC_MAJOR = 75;
export const ATTR_CC_MINOR = 76;

export function checkCU(label: string, code: number): void {
  if (code !== 0) throw new Error('CUDA driver error ' + code + ' in ' + label);
}

export function readProgramLog(prog: CudaHandle | null): string {
  const sz = [0n];
  nv.getProgramLogSize(prog, sz);
  const n = Number(sz[0]);
  if (n <= 1) return '';
  const buf = new Uint8Array(n);
  nv.getProgramLog(prog, buf);
  let s = '';
  for (const c of buf) { if (c === 0) break; s += String.fromCharCode(c); }
  return s;
}
