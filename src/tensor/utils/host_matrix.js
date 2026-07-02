import { tensor } from '../factory/from_ops.js';
import { tensorToContiguous } from '../../dispatcher/jit_dispatch.js';
import { DeviceType, CPU_DEVICE } from '../types/device.js';

const HOST_DEVICES = new Set([DeviceType.CPU, DeviceType.WASM]);

export function deviceKey(t) {
  return t.device.type;
}

function requireHost(t) {
  if (!HOST_DEVICES.has(t.device.type)) {
    throw new Error(`linalg/ml: host-readable device required (cpu or wasm), got '${t.device.type}'`);
  }
}

export function hostMatrix(t) {
  if (t.ndim !== 2) throw new Error(`linalg/ml: expected a 2-D matrix, got ${t.ndim}-D`);
  requireHost(t);
  const [rows, cols] = t.shape;
  return { data: Float64Array.from(tensorToContiguous(t)), rows, cols };
}

export function hostColumns(t) {
  requireHost(t);
  if (t.ndim === 1) return { data: Float64Array.from(tensorToContiguous(t)), rows: t.shape[0], cols: 1, wasVector: true };
  if (t.ndim === 2) return { data: Float64Array.from(tensorToContiguous(t)), rows: t.shape[0], cols: t.shape[1], wasVector: false };
  throw new Error(`linalg/ml: expected a 1-D or 2-D right-hand side, got ${t.ndim}-D`);
}

export function toHostTensor(data, shape, dtype, device = CPU_DEVICE) {
  return tensor(data, { shape, dtype, device });
}
