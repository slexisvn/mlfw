import { getDevice } from './cuda/device.js';
import { getProgram, getProgramFor } from './cuda/program.js';
import { acquire, copyHostToDevice, copyDeviceToHost, release, free, drainPool } from './cuda/memory.js';
import { launch, f32 } from './cuda/launcher.js';
import { runCudaPlan, setCudaGraphEnabled, isCudaGraphEnabled } from './cuda/device_plan.js';
import { uploadIfStale, downloadAndValidate, deviceBufferForInput, deviceBufferForOutput, deviceBufferForInplace, isEagerDeferred, hostReadHook, pinResident, releaseAllResident, flushDeferred } from './cuda/resident.js';
import { isEagerCapturing, isCudaGraphArmed, setCudaGraphArmed } from '../../dispatcher/eager_mode.js';
import { destroyEagerGraph } from './cuda/eager_graph.js';
import { cu } from './cuda/ffi.js';
import { StorageImpl } from '../../tensor/core/storage_impl.js';
import { registerMeasurer } from '../measurer_registry.js';
import { setGpuContiguousFn, setGpuConcatFn, setCudnnLSTM, setCudnnGRU, setGpuAdamFn, setGpuMatmul, wrapResult } from '../../dispatcher/jit_dispatch.js';
import { setGpuContiguousHook, contiguous as viewContiguous } from '../../tensor/view/view_ops.js';
import { IndexSelectBackward } from '../../autograd/function/indexing.js';
import { DeviceType } from '../../tensor/types/device.js';
import { typedArrayCtor } from '../../tensor/types/dtype.js';
import { cudnnAvailable } from './cuda/cudnn.js';
import { cudnnLSTMOp, cudnnGRUOp } from './cuda/cudnn_rnn_op.js';
import { gpuMatmul } from './cuda/cublas_matmul_op.js';
import { registerCudaLinalg } from '../../kernels/cuda/linalg/register.js';

export { runCudaPlan, setCudaGraphEnabled, isCudaGraphEnabled };

StorageImpl.setHostReadHook(hostReadHook);

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
  const smem = 0;
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
  const scratch = _acquireScratch(meta.scratch);

  launch(func, meta.gridDim, meta.blockDim, 0, [...ptrs, ...scratch.ptrs], scalars);

  const outputSet = new Set(outputs);
  for (let i = 0; i < buffers.length; i++) {
    if (outputSet.has(i)) copyDeviceToHost(buffers[i], ptrs[i]);
  }
  for (let i = 0; i < ptrs.length; i++) release(ptrs[i], buffers[i].byteLength);
  _releaseScratch(scratch);
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
    launch(func, meta.gridDim, meta.blockDim, 0, [...ptrs, ...scratch.ptrs], scalars, false);
    _releaseScratch(scratch);
    return;
  }

  for (let i = 0; i < buffers.length; i++) ptrs[i] = uploadIfStale(buffers[i]);
  launch(func, meta.gridDim, meta.blockDim, 0, [...ptrs, ...scratch.ptrs], scalars, false);
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

function _stepIncKernel(name) {
  return `extern "C" __global__ void ${name}(int* t) {
  if (threadIdx.x == 0 && blockIdx.x == 0) t[0] = t[0] + 1;
}`;
}

function _adamGraphKernel(ct, name) {
  return `extern "C" __global__ void ${name}(${ct}* w, const ${ct}* g, ${ct}* m, ${ct}* v, const int* t, int n, float b1, float b2, float ob1, float ob2, float eps, float lr, float wd) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
  int step = t[0];
  double bc1 = 1.0 - pow((double)b1, (double)step);
  double bc2 = 1.0 - pow((double)b2, (double)step);
  float ss = (float)((double)lr / bc1);
  float bc2s = (float)sqrt(bc2);
  float gi = (float)g[i] + wd * (float)w[i];
  float mi = b1*(float)m[i] + ob1*gi;
  float vi = b2*(float)v[i] + ob2*gi*gi;
  m[i] = (${ct})mi; v[i] = (${ct})vi;
  w[i] = (${ct})((float)w[i] - ss * mi / (sqrtf(vi)/bc2s + eps));
}`;
}

function _clipAccumKernel(name) {
  return `extern "C" __global__ void ${name}(const float* g, double* acc, int n) {
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  double v = (i < n) ? (double)g[i]*(double)g[i] : 0.0;
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads(); }
  if (tid == 0) atomicAdd(acc, sdata[0]);
}`;
}
function _clipCoefKernel(name) {
  return `extern "C" __global__ void ${name}(const double* acc, const float* mp, float* coef) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    double norm = sqrt(acc[0]);
    double c = (double)mp[0] / (norm + (double)mp[1]);
    coef[0] = c < 1.0 ? (float)c : 1.0f;
  }
}`;
}
function _clipScaleKernel(name) {
  return `extern "C" __global__ void ${name}(float* g, const float* coef, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) g[i] = g[i] * coef[0];
}`;
}

let _clipAcc = null, _clipCoef = null, _clipMax = null;
export function deviceClipGradNorm(params, maxNorm, eps = 1e-6) {
  getDevice();
  if (_clipAcc === null) { _clipAcc = acquire(8); _clipCoef = acquire(4); _clipMax = acquire(8); }
  const stream = getDevice().stream;
  if (!isEagerCapturing()) copyHostToDevice(_clipMax, new Float32Array([maxNorm, eps]));
  if (isEagerCapturing()) cu.memsetD8Async(_clipAcc, 0, 8, stream);
  else cu.memsetD8(_clipAcc, 0, 8);
  const accumF = getProgram(_clipAccumKernel('clip_accum'), 'clip_accum').func;
  const coefF = getProgram(_clipCoefKernel('clip_coef'), 'clip_coef').func;
  const scaleF = getProgram(_clipScaleKernel('clip_scale'), 'clip_scale').func;
  const grads = [];
  for (const p of params) {
    if (!p.grad) continue;
    const gRaw = p.grad._impl.storage.rawData;
    if (!gRaw) continue;
    const n = gRaw.length;
    const gDev = deviceBufferForInplace(gRaw);
    grads.push([gDev, n]);
    launch(accumF, [Math.ceil(n / 256), 1, 1], [256, 1, 1], 256 * 8, [gDev, _clipAcc], [n], false);
  }
  if (grads.length === 0) return;
  launch(coefF, [1, 1, 1], [1, 1, 1], 0, [_clipAcc, _clipMax, _clipCoef], [], false);
  for (const [gDev, n] of grads) {
    launch(scaleF, [Math.ceil(n / 256), 1, 1], [256, 1, 1], 0, [gDev, _clipCoef], [n], false);
  }
}

export function freeOptimizerDeviceState(optimizer) {
  const state = optimizer && optimizer._state;
  if (!state || typeof state.values !== 'function') return;
  for (const st of state.values()) {
    if (!st) continue;
    if (st._mDev !== undefined) { free(st._mDev); delete st._mDev; }
    if (st._vDev !== undefined) { free(st._vDev); delete st._vDev; }
    if (st._tDev !== undefined) { free(st._tDev); delete st._tDev; }
  }
}

function freeClipScratch() {
  if (_clipAcc !== null) {
    free(_clipAcc); free(_clipCoef); free(_clipMax);
    _clipAcc = null; _clipCoef = null; _clipMax = null;
  }
}

export function releaseCudaMemory() {
  releaseAllResident();
  freeClipScratch();
  return drainPool();
}

export function teardownAfterFit(model, optimizers) {
  const runner = model && model.__eagerGraphRunner;
  if (runner) {
    try { destroyEagerGraph(runner.captured); } catch (_) {}
    model._cudaGraphPhase = runner.phase;
    delete model.__eagerGraphRunner;
  }
  if (optimizers) for (const o of optimizers) freeOptimizerDeviceState(o);
  setCudaGraphArmed(false);
  flushDeferred();
  freeClipScratch();
  return drainPool();
}

export function stepIncKernelSource() {
  const name = 'step_inc';
  let src = _kernSrc.get(name); if (!src) { src = _stepIncKernel(name); _kernSrc.set(name, src); }
  return { source: src, name };
}

export function adamGraphKernelSource(dtype = 'f32') {
  const name = `adam_graph_${dtype}`;
  let src = _kernSrc.get(name); if (!src) { src = _adamGraphKernel(_ctype(dtype), name); _kernSrc.set(name, src); }
  return { source: src, name };
}

function _setDeviceInt(dptr, value) {
  copyHostToDevice(dptr, new Int32Array([value | 0]));
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
  const armed = isCudaGraphArmed();
  if (!state._mDev) {
    state._mDev = acquire(n * 4); cu.memsetD8(state._mDev, 0, n * 4);
    state._vDev = acquire(n * 4); cu.memsetD8(state._vDev, 0, n * 4);
    if (armed) { state._tDev = acquire(4); cu.memsetD8(state._tDev, 0, 4); }
    pinResident(wRaw);
  }
  const wDev = deviceBufferForInplace(wRaw);
  const gDev = deviceBufferForInput(gRaw);
  if (isEagerCapturing()) {
    if (state._tDev === undefined) throw new Error('CUDA graph Adam requires a device step counter; warmup must run before capture');
    const si = stepIncKernelSource();
    const ag = adamGraphKernelSource(dtype);
    const stepInc = getProgram(si.source, si.name).func;
    const adamG = getProgram(ag.source, ag.name).func;
    launch(stepInc, [1, 1, 1], [1, 1, 1], 0, [state._tDev], [], false);
    launch(adamG, [Math.ceil(n / 256), 1, 1], [256, 1, 1], 0,
      [wDev, gDev, state._mDev, state._vDev, state._tDev],
      [n, f32(sc.beta1), f32(sc.beta2), f32(sc.omb1), f32(sc.omb2), f32(sc.eps), f32(sc.lr), f32(sc.wd)], false);
    return true;
  }
  const name = `adam_${dtype}`;
  let src = _kernSrc.get(name);
  if (!src) { src = _adamKernel(_ctype(dtype), name); _kernSrc.set(name, src); }
  const { func } = getProgram(src, name);
  launch(func, [Math.ceil(n / 256), 1, 1], [256, 1, 1], 0,
    [wDev, gDev, state._mDev, state._vDev],
    [n, f32(sc.beta1), f32(sc.beta2), f32(sc.omb1), f32(sc.omb2), f32(sc.eps), f32(sc.stepSize), f32(sc.bc2sqrt), f32(sc.wd)], false);
  if (armed && state._tDev !== undefined) _setDeviceInt(state._tDev, state.step);
  return true;
}

function gpuContiguousTensor(t) {
  if (!isEagerDeferred() || !t.device || t.device.type !== DeviceType.GPU) return null;
  const out = deviceContiguous(t._impl.storage.rawData, t.shape, t.strides, t._impl.storageOffset, t.dtype);
  return wrapResult(out, [...t.shape], t.dtype, t.device);
}
setGpuContiguousHook(gpuContiguousTensor);

const _idxAddSrc = `extern "C" __global__ void index_add0(const float* g, const int* idx, float* out, int K, int E) {
  int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= K*E) return;
  int i = t / E, j = t % E;
  atomicAdd(&out[idx[i]*E + j], g[i*E + j]);
}`;
function gpuIndexSelectBackward(g, index, inputShape, dim) {
  if (!isEagerDeferred() || dim !== 0 || inputShape.length !== 2) return null;
  if (!g.device || g.device.type !== DeviceType.GPU) return null;
  if (index.dtype !== 'i32') return null;
  const V = inputShape[0], E = inputShape[1];
  const gc = viewContiguous(g), idxC = viewContiguous(index);
  const K = idxC.numel;
  if (gc.numel !== K * E) return null;
  getDevice();
  const out = new Float32Array(V * E);
  const { func } = getProgram(_idxAddSrc, 'index_add0');
  const outDev = deviceBufferForOutput(out);
  const stream = getDevice().stream;
  if (isEagerCapturing()) cu.memsetD8Async(outDev, 0, V * E * 4, stream);
  else cu.memsetD8(outDev, 0, V * E * 4);
  const gDev = deviceBufferForInput(gc._impl.storage.rawData);
  const idxDev = deviceBufferForInput(idxC._impl.storage.rawData);
  launch(func, [Math.ceil((K * E) / 256), 1, 1], [256, 1, 1], 0, [gDev, idxDev, outDev], [K, E], false);
  return wrapResult(out, [V, E], g.dtype, g.device);
}
IndexSelectBackward.setGpuBackward(gpuIndexSelectBackward);
setGpuContiguousFn(deviceContiguous);
setGpuConcatFn(deviceConcat);
setGpuAdamFn(deviceAdam);
setGpuMatmul(gpuMatmul);
if (cudnnAvailable()) { setCudnnLSTM(cudnnLSTMOp); setCudnnGRU(cudnnGRUOp); }
registerCudaLinalg();

registerMeasurer('cuda', measureCudaKernel);
