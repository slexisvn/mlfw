import koffi from 'koffi';
import { loadCudaLib, cudaIncludeDir, DRIVER_SPEC, NVRTC_SPEC } from './lib_resolver.js';

export { cudaIncludeDir };

const drv = koffi.load(loadCudaLib(DRIVER_SPEC));
const nvrtc = koffi.load(loadCudaLib(NVRTC_SPEC));

export const cu = {
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

export const nv = {
  createProgram: nvrtc.func('int nvrtcCreateProgram(_Out_ void **prog, str src, str name, int n, void *h, void *inc)'),
  compileProgram: nvrtc.func('int nvrtcCompileProgram(void *prog, int n, str *opts)'),
  destroyProgram: nvrtc.func('int nvrtcDestroyProgram(void **prog)'),
  getPTXSize: nvrtc.func('int nvrtcGetPTXSize(void *prog, _Out_ size_t *sz)'),
  getPTX: nvrtc.func('int nvrtcGetPTX(void *prog, _Out_ uint8_t *ptx)'),
  getProgramLogSize: nvrtc.func('int nvrtcGetProgramLogSize(void *prog, _Out_ size_t *sz)'),
  getProgramLog: nvrtc.func('int nvrtcGetProgramLog(void *prog, _Out_ uint8_t *log)'),
};

export const ATTR_CC_MAJOR = 75;
export const ATTR_CC_MINOR = 76;

export function checkCU(label, code) {
  if (code !== 0) throw new Error('CUDA driver error ' + code + ' in ' + label);
}

export function readProgramLog(prog) {
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
