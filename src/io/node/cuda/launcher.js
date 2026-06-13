import { cu, checkCU } from './ffi.js';
import { getDevice } from './device.js';

function devicePtrParam(dptr) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(dptr));
  return b;
}

function scalarParam(value) {
  const b = Buffer.alloc(4);
  if (Number.isInteger(value)) b.writeInt32LE(value | 0);
  else b.writeFloatLE(value);
  return b;
}

export function launch(func, gridDim, blockDim, sharedMemBytes, devicePtrs, scalars) {
  const { stream } = getDevice();
  const params = [];
  for (const p of devicePtrs) params.push(devicePtrParam(p));
  for (const s of scalars) params.push(scalarParam(s));
  checkCU('cuLaunchKernel', cu.launchKernel(
    func,
    gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2],
    sharedMemBytes, stream, params, null,
  ));
  checkCU('cuStreamSynchronize', cu.streamSynchronize(stream));
}
