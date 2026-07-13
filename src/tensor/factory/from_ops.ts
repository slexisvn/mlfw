import { Tensor } from '../core/tensor.js';
import { TensorImpl } from '../core/tensor_impl.js';
import { Storage } from '../core/storage.js';
import { ScalarType, typedArrayCtor } from '../types/dtype.js';
import type { DType, NumericTypedArray } from '../types/dtype.js';
import { getDefaultDevice } from '../types/device.js';
import type { Device } from '../types/device.js';
import { computeStrides } from '../utils/shape_utils.js';
import { coerceForStorage } from '../utils/half.js';

type TensorOptions = {
  shape?: readonly number[];
  dtype?: DType;
  device?: Device;
  requiresGrad?: boolean;
  offset?: number;
};
type NestedNumberArray = readonly (number | NestedNumberArray)[];
type TensorInput = number | ArrayLike<number> | NestedNumberArray | NumericTypedArray;
type MutableNumericArray = { length: number; [index: number]: number | bigint };

export function tensor(data: TensorInput, opts?: TensorOptions): Tensor {
  const dtype = opts?.dtype ?? ScalarType.F32;
  const device = opts?.device ?? getDefaultDevice();
  const requiresGrad = opts?.requiresGrad ?? false;

  if (ArrayBuffer.isView(data) && 'length' in data) {
    return _fromTypedArray(data as ArrayLike<number>, opts?.shape, dtype, device, requiresGrad);
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

export function fromBuffer(buffer: NumericTypedArray, shape: readonly number[], dtype: DType, opts?: TensorOptions): Tensor {
  const device = opts?.device ?? getDefaultDevice();
  const requiresGrad = opts?.requiresGrad ?? false;
  const strides = computeStrides(shape);
  const storage = Storage.fromData(buffer, device);
  const offset = opts?.offset ?? 0;
  const impl = new TensorImpl(storage, offset, shape, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

export function scalar(value: number, opts?: TensorOptions): Tensor {
  const dtype = opts?.dtype ?? ScalarType.F32;
  const device = opts?.device ?? getDefaultDevice();
  return _fromScalar(value, dtype, device, opts?.requiresGrad ?? false);
}

const _NEEDS_COERCE = new Set(['f16', 'bf16', 'i64']);

function _fromScalar(value: number, dtype: DType, device: Device, requiresGrad: boolean): Tensor {
  const Ctor = typedArrayCtor(dtype);
  const data = new Ctor(1);
  const writable = data as MutableNumericArray;
  writable[0] = _NEEDS_COERCE.has(dtype) ? coerceForStorage(dtype, value) : value;
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, [], [], dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

function _fromTypedArray(data: ArrayLike<number>, shape: readonly number[] | undefined, dtype: DType, device: Device, requiresGrad: boolean): Tensor {
  const Ctor = typedArrayCtor(dtype);
  const copied = new Ctor(data.length);
  const writable = copied as MutableNumericArray;
  if (_NEEDS_COERCE.has(dtype)) {
    for (let i = 0; i < data.length; i++) writable[i] = coerceForStorage(dtype, data[i]!);
  } else {
    for (let i = 0; i < data.length; i++) writable[i] = data[i]!;
  }

  const finalShape = shape ?? [data.length];
  const strides = computeStrides(finalShape);
  const storage = Storage.fromData(copied, device);
  const impl = new TensorImpl(storage, 0, finalShape, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

function _fromFlatArray(flat: readonly number[], shape: readonly number[], dtype: DType, device: Device, requiresGrad: boolean): Tensor {
  const Ctor = typedArrayCtor(dtype);
  const data = new Ctor(flat.length);
  const writable = data as MutableNumericArray;
  if (_NEEDS_COERCE.has(dtype)) {
    for (let i = 0; i < flat.length; i++) writable[i] = coerceForStorage(dtype, flat[i]!);
  } else {
    for (let i = 0; i < flat.length; i++) writable[i] = flat[i]!;
  }

  const strides = computeStrides(shape);
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, device);
  const t = new Tensor(impl);
  if (requiresGrad) t.requiresGrad_(true);
  return t;
}

function _flattenNested(arr: NestedNumberArray): { flat: number[]; shape: number[] } {
  const shape: number[] = [];
  let current: number | NestedNumberArray | undefined = arr;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  const flat: number[] = [];
  _recurFlatten(arr, flat, shape, 0);
  return { flat, shape };
}

function _recurFlatten(arr: NestedNumberArray, flat: number[], shape: readonly number[], dim: number) {
  if (dim === shape.length - 1) {
    for (let i = 0; i < arr.length; i++) flat.push(arr[i] as number);
    return;
  }
  for (let i = 0; i < arr.length; i++) {
    _recurFlatten(arr[i] as NestedNumberArray, flat, shape, dim + 1);
  }
}
