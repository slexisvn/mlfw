export function runCudaKernel(): never {
  throw new Error('CUDA runtime is not available in the browser');
}

export function measureCudaKernel(): never {
  throw new Error('CUDA runtime is not available in the browser');
}

export function runCudaPlan(): never {
  throw new Error('CUDA runtime is not available in the browser');
}

export function deviceClipGradNorm(): never {
  throw new Error('CUDA runtime is not available in the browser');
}

export function releaseCudaMemory(): number {
  return 0;
}
