import { getDevice } from './cuda/device.js';
import { getProgram, getProgramFor } from './cuda/program.js';
import { acquire, copyHostToDevice, copyDeviceToHost, release } from './cuda/memory.js';
import { launch } from './cuda/launcher.js';
import { runCudaPlan, setCudaGraphEnabled, isCudaGraphEnabled } from './cuda/device_plan.js';
import { uploadIfStale, downloadAndValidate, deviceBufferForInput, deviceBufferForOutput, deviceBufferForInplace, isEagerDeferred, hostReadHook, pinResident } from './cuda/resident.js';
import { cu } from './cuda/ffi.js';
import { StorageImpl } from '../../tensor/core/storage_impl.js';
import { registerMeasurer } from '../../compiler/runtime/measurer_registry.js';
import { setGpuContiguousFn, setGpuConcatFn, setCudnnLSTM, setGpuAdamFn, setGpuMatmul } from '../../dispatcher/jit_dispatch.js';
import { typedArrayCtor } from '../../tensor/types/dtype.js';
import { cudnnAvailable } from './cuda/cudnn.js';
import { cudnnLSTMOp } from './cuda/cudnn_lstm_op.js';
import { gpuMatmul } from './cuda/cublas_matmul_op.js';

export { runCudaPlan, setCudaGraphEnabled, isCudaGraphEnabled };

StorageImpl._hostReadHook = hostReadHook;

export function measureCudaKernel(compiledKernel, bufferByteSizes, shapeValues = [], opts = {}) {
  getDevice();
  const { func } = getProgram(compiledKernel.source, compiledKernel.name);
  const meta = compiledKernel.metadata;
  const warmup = opts.warmup ?? 5;
  const repeat = opts.repeat ?? 30;
  const minWarmupMs = opts.minWarmupMs ?? 25;
  const maxWarmup = opts.maxWarmup ?? 100000;
  const grid = meta.gridDim;
  const block = meta.blockDim;
  const smem = meta.sharedMemBytes || 0;
  const sizes = bufferByteSizes.map(bytes => Math.max(bytes, 1));
  const ptrs = sizes.map(bytes => acquire(bytes));
  try {
    const wStart = performance.now();
    let w = 0;
    while (w < warmup || (performance.now() - wStart) < minWarmupMs) {
      launch(func, grid, block, smem, ptrs, shapeValues);
      if (++w >= maxWarmup) break;
    }
    const samples = [];
    for (let i = 0; i < repeat; i++) {
      const t0 = performance.now();
      launch(func, grid, block, smem, ptrs, shapeValues);
      samples.push(performance.now() - t0);
    }
    return samples;
  } finally {
    for (let i = 0; i < ptrs.length; i++) release(ptrs[i], sizes[i]);
  }
}

let _cublas = null;

export async function preloadCublas() {
  if (!_cublas) _cublas = await import('./cuda/cublas.js');
  return _cublas;
}

export function runCudaKernelSync(compiledKernel, tensorArgs, shapeValues) {
  const meta = compiledKernel.metadata;
  if (meta.cublas) {
    if (!_cublas) throw new Error('cuBLAS module not preloaded; call preloadCublas() before sync execution');
    const { M, N, K, aIdx, bIdx, cIdx, transB } = meta.cublas;
    _cublas.cublasMatmul(M, N, K, tensorArgs[aIdx], tensorArgs[bIdx], tensorArgs[cIdx], transB);
    return;
  }
  getDevice();
  const { func } = getProgramFor(compiledKernel);

  const buffers = [];
  const scalars = [];
  for (const a of tensorArgs) {
    if (ArrayBuffer.isView(a)) buffers.push(a);
    else scalars.push(a);
  }
  if (shapeValues) for (const v of shapeValues) scalars.push(v);

  const outputs = meta.outputIndices || buffers.map((_, i) => i);
  const ptrs = [];
  for (const host of buffers) {
    const dptr = acquire(host.byteLength);
    copyHostToDevice(dptr, host);
    ptrs.push(dptr);
  }

  launch(func, meta.gridDim, meta.blockDim, meta.sharedMemBytes || 0, ptrs, scalars);

  const outputSet = new Set(outputs);
  for (let i = 0; i < buffers.length; i++) {
    if (outputSet.has(i)) copyDeviceToHost(buffers[i], ptrs[i]);
  }
  for (let i = 0; i < ptrs.length; i++) release(ptrs[i], buffers[i].byteLength);
}

export function runCudaKernelResident(compiledKernel, tensorArgs, shapeValues) {
  const meta = compiledKernel.metadata;
  if (meta.cublas) throw new Error('cuBLAS kernels are not supported on the eager device-resident path');
  getDevice();
  const { func } = getProgramFor(compiledKernel);

  const buffers = [];
  const scalars = [];
  for (const a of tensorArgs) {
    if (ArrayBuffer.isView(a)) buffers.push(a);
    else scalars.push(a);
  }
  if (shapeValues) for (const v of shapeValues) scalars.push(v);

  const outputSet = meta._outputSet || (meta._outputSet = new Set(meta.outputIndices || buffers.map((_, i) => i)));
  const ptrs = new Array(buffers.length);
  const scratch = _acquireScratch(meta.scratch);

  if (isEagerDeferred()) {
    for (let i = 0; i < buffers.length; i++) {
      ptrs[i] = outputSet.has(i) ? deviceBufferForOutput(buffers[i]) : deviceBufferForInput(buffers[i]);
    }
    launch(func, meta.gridDim, meta.blockDim, meta.sharedMemBytes || 0, [...ptrs, ...scratch.ptrs], scalars, false);
    _releaseScratch(scratch);
    return;
  }

  for (let i = 0; i < buffers.length; i++) ptrs[i] = uploadIfStale(buffers[i]);
  launch(func, meta.gridDim, meta.blockDim, meta.sharedMemBytes || 0, [...ptrs, ...scratch.ptrs], scalars, false);
  for (let i = 0; i < buffers.length; i++) {
    if (outputSet.has(i)) downloadAndValidate(buffers[i], ptrs[i]);
  }
  _releaseScratch(scratch);
}

function _acquireScratch(scratch) {
  if (!scratch || scratch.length === 0) return { ptrs: [], bufs: [] };
  const ptrs = [], bufs = [];
  for (const s of scratch) {
    const bytes = s.size * typedArrayCtor(s.dtype).BYTES_PER_ELEMENT;
    const p = acquire(bytes);
    ptrs.push(p); bufs.push([p, bytes]);
  }
  return { ptrs, bufs };
}
function _releaseScratch(scratch) {
  for (const [p, bytes] of scratch.bufs) release(p, bytes);
}

export async function runCudaKernel(compiledKernel, tensorArgs, shapeValues) {
  if (compiledKernel.metadata.cublas) await preloadCublas();
  runCudaKernelSync(compiledKernel, tensorArgs, shapeValues);
}

const _CTYPE = { f32: 'float', f64: 'double', i64: 'long long', i32: 'int', i16: 'short', i8: 'signed char', u8: 'unsigned char', bool: 'unsigned char' };
function _ctype(dtype) { return _CTYPE[dtype] || 'float'; }

const _kernSrc = new Map();
function _meta(n, outIdx) { return { gridDim: [Math.max(Math.ceil(n / 256), 1), 1, 1], blockDim: [256, 1, 1], sharedMemBytes: 0, outputIndices: outIdx }; }

function _gatherKernel(ct, name) {
  return `extern "C" __global__ void ${name}(const ${ct}* in, ${ct}* out, int n, int rank,
  int s0,int s1,int s2,int s3,int s4,int s5,int s6,int s7,
  int t0,int t1,int t2,int t3,int t4,int t5,int t6,int t7, int off) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
  int shp[8] = {s0,s1,s2,s3,s4,s5,s6,s7};
  int strd[8] = {t0,t1,t2,t3,t4,t5,t6,t7};
  long long src = off; int rem = i;
  for (int d = rank-1; d >= 0; d--) { int x = rem % shp[d]; rem /= shp[d]; src += (long long)x*strd[d]; }
  out[i] = in[src];
}`;
}

export function deviceContiguous(rawData, shape, strides, offset, dtype) {
  const ct = _ctype(dtype);
  const name = `gather_${dtype}`;
  let src = _kernSrc.get(name); if (!src) { src = _gatherKernel(ct, name); _kernSrc.set(name, src); }
  const rank = shape.length;
  let n = 1; for (let i = 0; i < rank; i++) n *= shape[i];
  const out = new (typedArrayCtor(dtype))(Math.max(n, 1));
  const shp = new Array(8).fill(1), strd = new Array(8).fill(0);
  for (let i = 0; i < rank; i++) { shp[i] = shape[i]; strd[i] = strides[i]; }
  runCudaKernelResident({ source: src, name, metadata: _meta(n, [1]) }, [rawData, out, n, rank, ...shp, ...strd, offset | 0], null);
  return out;
}

function _catKernel(ct, name) {
  return `extern "C" __global__ void ${name}(const ${ct}* in, ${ct}* out, int pre, int dk, int tail, int total, int offset) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; int n = pre*dk*tail; if (i >= n) return;
  int t = i % tail; int r = i / tail; int j = r % dk; int p = r / dk;
  out[(long long)p*total*tail + (long long)(offset+j)*tail + t] = in[i];
}`;
}

export function deviceConcat(opName, inputArrays, inputShapes, dim, outShape, outData, dtype) {
  const ct = _ctype(dtype);
  const name = `catcopy_${dtype}`;
  let src = _kernSrc.get(name); if (!src) { src = _catKernel(ct, name); _kernSrc.set(name, src); }
  const isStack = opName === 'stack';
  const rank = outShape.length;
  const d = dim < 0 ? rank + dim : dim;
  let pre = 1; for (let i = 0; i < d; i++) pre *= outShape[i];
  let tail = 1; for (let i = d + 1; i < rank; i++) tail *= outShape[i];
  const total = outShape[d];
  let offset = 0;
  for (let k = 0; k < inputArrays.length; k++) {
    const dk = isStack ? 1 : inputShapes[k][d];
    const n = pre * dk * tail;
    runCudaKernelResident({ source: src, name, metadata: _meta(n, [1]) }, [inputArrays[k], outData, pre, dk, tail, total, offset], null);
    offset += dk;
  }
}

function _adamKernel(ct, name) {
  return `extern "C" __global__ void ${name}(${ct}* w, const ${ct}* g, ${ct}* m, ${ct}* v, int n, float b1, float b2, float ob1, float ob2, float eps, float ss, float bc2s, float wd) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
  float gi = (float)g[i] + wd * (float)w[i];
  float mi = b1*(float)m[i] + ob1*gi;
  float vi = b2*(float)v[i] + ob2*gi*gi;
  m[i] = (${ct})mi; v[i] = (${ct})vi;
  w[i] = (${ct})((float)w[i] - ss * mi / (sqrtf(vi)/bc2s + eps));
}`;
}

export function deviceAdam(p, state, sc) {
  if (!isEagerDeferred()) return false;
  const dtype = p.dtype || 'f32';
  if (dtype !== 'f32') return false;
  const wRaw = p._impl.storage.rawData;
  const gRaw = p.grad._impl.storage.rawData;
  if (!wRaw || !gRaw) return false;
  const n = wRaw.length;
  getDevice();
  const name = `adam_${dtype}`;
  let src = _kernSrc.get(name);
  if (!src) { src = _adamKernel(_ctype(dtype), name); _kernSrc.set(name, src); }
  const { func } = getProgram(src, name);
  if (!state._mDev) {
    state._mDev = acquire(n * 4); cu.memsetD8(state._mDev, 0, n * 4);
    state._vDev = acquire(n * 4); cu.memsetD8(state._vDev, 0, n * 4);
    pinResident(wRaw);
  }
  const wDev = deviceBufferForInplace(wRaw);
  const gDev = deviceBufferForInput(gRaw);
  launch(func, [Math.ceil(n / 256), 1, 1], [256, 1, 1], 0,
    [wDev, gDev, state._mDev, state._vDev],
    [n, sc.beta1, sc.beta2, sc.omb1, sc.omb2, sc.eps, sc.stepSize, sc.bc2sqrt, sc.wd], false);
  return true;
}

setGpuContiguousFn(deviceContiguous);
setGpuConcatFn(deviceConcat);
setGpuAdamFn(deviceAdam);
setGpuMatmul(gpuMatmul);
if (cudnnAvailable()) setCudnnLSTM(cudnnLSTMOp);

registerMeasurer('cuda', measureCudaKernel);
