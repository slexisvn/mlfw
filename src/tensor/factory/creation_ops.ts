import { Tensor } from '../core/tensor.js';
import { random } from '../../util/random.js';
import { TensorImpl } from '../core/tensor_impl.js';
import { Storage } from '../core/storage.js';

import { ScalarType, dtypeSize } from '../types/dtype.js';
import type { DType, NumericTypedArray } from '../types/dtype.js';
import { getDefaultDevice } from '../types/device.js';
import type { Device } from '../types/device.js';
import type { MutableNumericArray, TensorOptions } from '../types/options.js';
import { computeStrides, computeNumel } from '../utils/shape_utils.js';

type NormalizedTensorOptions = {
  dtype: DType;
  device: Device;
  requiresGrad: boolean;
};

function _defaultOpts(opts?: TensorOptions): NormalizedTensorOptions {
  return {
    dtype: opts?.dtype ?? ScalarType.F32,
    device: opts?.device ?? getDefaultDevice(),
    requiresGrad: opts?.requiresGrad ?? false,
  };
}

function _makeTensor(sizes: readonly number[], dtype: DType, device: Device, requiresGrad: boolean): Tensor {
  const strides = computeStrides(sizes);
  const numel = computeNumel(sizes);
  const nbytes = numel * dtypeSize(dtype);
  const storage = Storage.allocate(nbytes, dtype, device);
  const impl = new TensorImpl(storage, 0, sizes, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

export function empty(shape: readonly number[], opts?: TensorOptions): Tensor {
  const { dtype, device, requiresGrad } = _defaultOpts(opts);
  return _makeTensor(shape, dtype, device, requiresGrad);
}

function _filled(shape: readonly number[], opts: TensorOptions | undefined, value: number | bigint): Tensor {
  const t = empty(shape, opts);
  const data = t.data as MutableNumericArray | null;
  if (data) { data.fill(value); return t; }
  if (t.device && t.device.type === 'meta') {
    const r = empty(shape, { dtype: opts?.dtype });
    (r.data as MutableNumericArray).fill(value);
    return r;
  }
  return t;
}

export function zeros(shape: readonly number[], opts?: TensorOptions): Tensor { return _filled(shape, opts, 0); }

export function ones(shape: readonly number[], opts?: TensorOptions): Tensor { return _filled(shape, opts, 1); }

export function full(shape: readonly number[], value: number | bigint, opts?: TensorOptions): Tensor { return _filled(shape, opts, value); }

export function randn(shape: readonly number[], opts?: TensorOptions): Tensor {
  const t = empty(shape, opts);
  const data = t.data;
  if (data) {
    const len = data.length;
    for (let i = 0; i < len; i += 2) {
      const u1 = random() || 1e-10;
      const u2 = random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 6.283185307179586 * u2;
      data[i] = r * Math.cos(theta);
      if (i + 1 < len) data[i + 1] = r * Math.sin(theta);
    }
  }
  return t;
}

export function arange(start: number, end?: number, step?: number, opts?: TensorOptions): Tensor {
  let s: number, e: number, st: number;
  if (end === undefined && step === undefined) {
    s = 0; e = start; st = 1;
  } else if (step === undefined) {
    s = start; e = end as number; st = 1;
  } else {
    s = start; e = end as number; st = step;
  }

  const len = Math.max(0, Math.ceil((e - s) / st));
  const { dtype, device, requiresGrad } = _defaultOpts(opts);
  const t = _makeTensor([len], dtype, device, requiresGrad);
  const data = t.data;
  if (data) {
    for (let i = 0; i < len; i++) data[i] = s + i * st;
  }
  return t;
}

export function eye(n: number, m?: number, opts?: TensorOptions): Tensor {
  const cols = m ?? n;
  const { dtype, device, requiresGrad } = _defaultOpts(opts);
  const t = _makeTensor([n, cols], dtype, device, requiresGrad);
  const data = t.data;
  if (data) {
    (data as MutableNumericArray).fill(0);
    const minDim = Math.min(n, cols);
    for (let i = 0; i < minDim; i++) data[i * cols + i] = 1;
  }
  return t;
}

export function randperm(n: number, opts?: TensorOptions): Tensor {
  const dtype = opts?.dtype ?? ScalarType.I32;
  const device = opts?.device ?? getDefaultDevice();
  const t = _makeTensor([n], dtype, device, opts?.requiresGrad ?? false);
  const data = t.data as NumericTypedArray;
  for (let i = 0; i < n; i++) data[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
  return t;
}

export function linspace(start: number, end: number, steps: number, opts?: TensorOptions): Tensor {
  const { dtype, device, requiresGrad } = _defaultOpts(opts);
  const t = _makeTensor([steps], dtype, device, requiresGrad);
  const data = t.data;
  if (data && steps > 0) {
    if (steps === 1) {
      data[0] = start;
    } else {
      const step = (end - start) / (steps - 1);
      for (let i = 0; i < steps; i++) data[i] = start + i * step;
    }
  }
  return t;
}
