import { tensor } from '../factory/from_ops.js';
import { float64From } from '../../util/numeric_array.js';
import { tensorToContiguous } from '../../dispatcher/jit_dispatch.js';
import { DeviceType, CPU_DEVICE } from '../types/device.js';
import type { Device } from '../types/device.js';
import type { DType, NumericTypedArray } from '../types/dtype.js';
import type { Tensor } from '../core/tensor.js';

const HOST_DEVICES = new Set<string>([DeviceType.CPU, DeviceType.WASM]);

export function deviceKey(t: Tensor): string {
  return t.device.type;
}

function requireHost(t: Tensor) {
  if (!HOST_DEVICES.has(t.device.type)) {
    throw new Error(`linalg/ml: host-readable device required (cpu or wasm), got '${t.device.type}'`);
  }
}

export type HostMatrix = { data: Float64Array; rows: number; cols: number };
export type HostColumns = HostMatrix & { wasVector: boolean };

export function hostMatrix(t: Tensor): HostMatrix {
  if (t.ndim !== 2) throw new Error(`linalg/ml: expected a 2-D matrix, got ${t.ndim}-D`);
  requireHost(t);
  const [rows, cols] = t.shape;
  return { data: float64From(tensorToContiguous(t)), rows, cols };
}

export function hostColumns(t: Tensor): HostColumns {
  requireHost(t);
  if (t.ndim === 1) return { data: float64From(tensorToContiguous(t)), rows: t.shape[0], cols: 1, wasVector: true };
  if (t.ndim === 2) return { data: float64From(tensorToContiguous(t)), rows: t.shape[0], cols: t.shape[1], wasVector: false };
  throw new Error(`linalg/ml: expected a 1-D or 2-D right-hand side, got ${t.ndim}-D`);
}

export function toHostTensor(data: NumericTypedArray | ArrayLike<number> | number[], shape: readonly number[], dtype: DType, device: Device = CPU_DEVICE): Tensor {
  return tensor(data, { shape, dtype, device });
}
