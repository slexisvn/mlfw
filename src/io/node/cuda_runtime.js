import { getDevice } from './cuda/device.js';
import { getProgram } from './cuda/program.js';
import { alloc, copyHostToDevice, copyDeviceToHost, free } from './cuda/memory.js';
import { launch } from './cuda/launcher.js';

export function instantiateCuda(compiledKernel) {
  getDevice();
  const program = getProgram(compiledKernel.source, compiledKernel.name);
  return { kernel: compiledKernel, program };
}

export async function runCudaKernel(compiledKernel, tensorArgs, shapeValues) {
  getDevice();
  const { func } = getProgram(compiledKernel.source, compiledKernel.name);
  const meta = compiledKernel.metadata;

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
    const dptr = alloc(host.byteLength);
    copyHostToDevice(dptr, host);
    ptrs.push(dptr);
  }

  launch(func, meta.gridDim, meta.blockDim, meta.sharedMemBytes || 0, ptrs, scalars);

  const outputSet = new Set(outputs);
  for (let i = 0; i < buffers.length; i++) {
    if (outputSet.has(i)) copyDeviceToHost(buffers[i], ptrs[i]);
  }
  for (const dptr of ptrs) free(dptr);
}
