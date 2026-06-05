import { Tensor } from '../core/tensor.js';
import { TensorImpl } from '../core/tensor_impl.js';
import { Storage } from '../core/storage.js';
import { ScalarType, typedArrayCtor, dtypeSize } from '../types/dtype.js';
import { CPU_DEVICE } from '../types/device.js';
import { computeStrides, computeNumel } from '../utils/shape_utils.js';

export function tensor(data, opts) {
  const dtype = opts?.dtype ?? ScalarType.F32;
  const device = opts?.device ?? CPU_DEVICE;
  const requiresGrad = opts?.requiresGrad ?? false;

  if (ArrayBuffer.isView(data)) {
    return _fromTypedArray(data, opts?.shape, dtype, device, requiresGrad);
  }

  if (Array.isArray(data)) {
    const { flat, shape } = _flattenNested(data);
    const finalShape = opts?.shape ?? shape;
    return _fromFlatArray(flat, finalShape, dtype, device, requiresGrad);
  }

  if (typeof data === 'number') {
    return _fromScalar(data, dtype, device, requiresGrad);
  }

  throw new Error('Unsupported data type for tensor()');
}

export function fromBuffer(buffer, shape, dtype, opts) {
  const device = opts?.device ?? CPU_DEVICE;
  const requiresGrad = opts?.requiresGrad ?? false;
  const strides = computeStrides(shape);
  const storage = Storage.fromData(buffer, device);
  const offset = opts?.offset ?? 0;
  const impl = new TensorImpl(storage, offset, shape, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

export function scalar(value, opts) {
  const dtype = opts?.dtype ?? ScalarType.F32;
  const device = opts?.device ?? CPU_DEVICE;
  return _fromScalar(value, dtype, device, opts?.requiresGrad ?? false);
}

function _fromScalar(value, dtype, device, requiresGrad) {
  const Ctor = typedArrayCtor(dtype);
  const data = new Ctor(1);
  data[0] = value;
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, [], [], dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

function _fromTypedArray(data, shape, dtype, device, requiresGrad) {
  const Ctor = typedArrayCtor(dtype);
  const copied = new Ctor(data.length);
  for (let i = 0; i < data.length; i++) copied[i] = data[i];

  const finalShape = shape ?? [data.length];
  const strides = computeStrides(finalShape);
  const storage = Storage.fromData(copied, device);
  const impl = new TensorImpl(storage, 0, finalShape, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

function _fromFlatArray(flat, shape, dtype, device, requiresGrad) {
  const Ctor = typedArrayCtor(dtype);
  const data = new Ctor(flat.length);
  for (let i = 0; i < flat.length; i++) data[i] = flat[i];

  const strides = computeStrides(shape);
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

function _flattenNested(arr) {
  const shape = [];
  let current = arr;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  const flat = [];
  _recurFlatten(arr, flat, shape, 0);
  return { flat, shape };
}

function _recurFlatten(arr, flat, shape, dim) {
  if (dim === shape.length - 1) {
    for (let i = 0; i < arr.length; i++) flat.push(arr[i]);
    return;
  }
  for (let i = 0; i < arr.length; i++) {
    _recurFlatten(arr[i], flat, shape, dim + 1);
  }
}
