import { cu, checkCU } from './ffi.js';
import type { CudaHandle, DevicePtr } from './ffi.js';
import { getDevice } from './device.js';
import { isEagerCapturing } from '../../../dispatcher/eager_mode.js';

export function devicePtrParam(dptr: DevicePtr | number | null | undefined): Buffer {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(dptr === null || dptr === undefined ? 0n : BigInt(dptr));
  return b;
}

class TypedScalar {
  kind: string;
  value: number;

  constructor(kind: string, value: number) { this.kind = kind; this.value = value; }
}

export function f32(value: number): TypedScalar { return new TypedScalar('f32', value); }
export function i32(value: number): TypedScalar { return new TypedScalar('i32', value); }

export function scalarParam(value: number | TypedScalar): Buffer {
  const b = Buffer.alloc(4);
  if (value instanceof TypedScalar) {
    if (value.kind === 'f32') b.writeFloatLE(value.value);
    else b.writeInt32LE(value.value | 0);
    return b;
  }
  if (Number.isInteger(value)) b.writeInt32LE(value | 0);
  else b.writeFloatLE(value);
  return b;
}

export function launch(
  func: CudaHandle | null,
  gridDim: readonly number[],
  blockDim: readonly number[],
  sharedMemBytes: number,
  devicePtrs: readonly (DevicePtr | number | null | undefined)[],
  scalars: readonly (number | TypedScalar)[],
  sync = true,
): void {
  const { stream } = getDevice();
  const params: Buffer[] = [];
  for (const p of devicePtrs) params.push(devicePtrParam(p));
  for (const s of scalars) params.push(scalarParam(s));
  checkCU('cuLaunchKernel', cu.launchKernel(
    func,
    gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2],
    sharedMemBytes, stream, params, null,
  ));
  if (sync && !isEagerCapturing()) checkCU('cuStreamSynchronize', cu.streamSynchronize(stream));
}
