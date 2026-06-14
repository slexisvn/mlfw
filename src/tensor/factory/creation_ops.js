import { Tensor } from '../core/tensor.js';
import { TensorImpl } from '../core/tensor_impl.js';
import { Storage } from '../core/storage.js';

import { ScalarType, dtypeSize } from '../types/dtype.js';
import { getDefaultDevice } from '../types/device.js';
import { computeStrides, computeNumel } from '../utils/shape_utils.js';

function _defaultOpts(opts) {
  return {
    dtype: opts?.dtype ?? ScalarType.F32,
    device: opts?.device ?? getDefaultDevice(),
    requiresGrad: opts?.requiresGrad ?? false,
  };
}

function _makeTensor(sizes, dtype, device, requiresGrad) {
  const strides = computeStrides(sizes);
  const numel = computeNumel(sizes);
  const nbytes = numel * dtypeSize(dtype);
  const storage = Storage.allocate(nbytes, dtype, device);
  const impl = new TensorImpl(storage, 0, sizes, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

export function empty(shape, opts) {
  const { dtype, device, requiresGrad } = _defaultOpts(opts);
  return _makeTensor(shape, dtype, device, requiresGrad);
}

export function zeros(shape, opts) {
  const t = empty(shape, opts);
  const data = t.data;
  if (data) data.fill(0);
  return t;
}

export function ones(shape, opts) {
  const t = empty(shape, opts);
  const data = t.data;
  if (data) data.fill(1);
  return t;
}

export function full(shape, value, opts) {
  const t = empty(shape, opts);
  const data = t.data;
  if (data) data.fill(value);
  return t;
}

export function randn(shape, opts) {
  const t = empty(shape, opts);
  const data = t.data;
  if (data) {
    const len = data.length;
    for (let i = 0; i < len; i += 2) {
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 6.283185307179586 * u2;
      data[i] = r * Math.cos(theta);
      if (i + 1 < len) data[i + 1] = r * Math.sin(theta);
    }
  }
  return t;
}

export function arange(start, end, step, opts) {
  let s, e, st;
  if (end === undefined && step === undefined) {
    s = 0; e = start; st = 1;
  } else if (step === undefined) {
    s = start; e = end; st = 1;
  } else {
    s = start; e = end; st = step;
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

export function eye(n, m, opts) {
  const cols = m ?? n;
  const { dtype, device, requiresGrad } = _defaultOpts(opts);
  const t = _makeTensor([n, cols], dtype, device, requiresGrad);
  const data = t.data;
  if (data) {
    data.fill(0);
    const minDim = Math.min(n, cols);
    for (let i = 0; i < minDim; i++) data[i * cols + i] = 1;
  }
  return t;
}

export function randperm(n, opts) {
  const dtype = opts?.dtype ?? ScalarType.I32;
  const device = opts?.device ?? getDefaultDevice();
  const t = _makeTensor([n], dtype, device, opts?.requiresGrad ?? false);
  const data = t.data;
  for (let i = 0; i < n; i++) data[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
  return t;
}

export function linspace(start, end, steps, opts) {
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
