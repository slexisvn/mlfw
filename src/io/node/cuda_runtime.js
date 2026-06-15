import { getDevice } from './cuda/device.js';
import { getProgram } from './cuda/program.js';
import { acquire, copyHostToDevice, copyDeviceToHost, release } from './cuda/memory.js';
import { launch } from './cuda/launcher.js';
import { runCudaPlan } from './cuda/device_plan.js';
import { uploadIfStale, downloadAndValidate, invalidate } from './cuda/resident.js';
import { StorageImpl } from '../../tensor/core/storage_impl.js';
import { registerMeasurer } from '../../compiler/runtime/measurer_registry.js';

export { runCudaPlan };

StorageImpl._hostReadHook = invalidate;

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
  const { func } = getProgram(compiledKernel.source, compiledKernel.name);

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
  const { func } = getProgram(compiledKernel.source, compiledKernel.name);

  const buffers = [];
  const scalars = [];
  for (const a of tensorArgs) {
    if (ArrayBuffer.isView(a)) buffers.push(a);
    else scalars.push(a);
  }
  if (shapeValues) for (const v of shapeValues) scalars.push(v);

  const outputSet = new Set(meta.outputIndices || buffers.map((_, i) => i));
  const ptrs = new Array(buffers.length);
  for (let i = 0; i < buffers.length; i++) ptrs[i] = uploadIfStale(buffers[i]);

  launch(func, meta.gridDim, meta.blockDim, meta.sharedMemBytes || 0, ptrs, scalars, false);

  for (let i = 0; i < buffers.length; i++) {
    if (outputSet.has(i)) downloadAndValidate(buffers[i], ptrs[i]);
  }
}

export async function runCudaKernel(compiledKernel, tensorArgs, shapeValues) {
  if (compiledKernel.metadata.cublas) await preloadCublas();
  runCudaKernelSync(compiledKernel, tensorArgs, shapeValues);
}

registerMeasurer('cuda', measureCudaKernel);
