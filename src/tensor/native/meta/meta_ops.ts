import { Tensor } from '../../core/tensor.js';
import { TensorImpl } from '../../core/tensor_impl.js';
import { Storage } from '../../core/storage.js';
import { META_DEVICE } from '../../types/device.js';
import { broadcastShapes, computeStrides, matmulOutputShape } from '../../utils/shape_utils.js';
import { resultDtype } from '../../types/dtype.js';
import type { DType } from '../../types/dtype.js';
import type { NativeKernel } from '../types.js';

function _metaTensor(shape: readonly number[], dtype: DType): Tensor {
  const strides = computeStrides(shape);
  const storage = Storage.allocate(0, dtype, META_DEVICE);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, META_DEVICE);
  return new Tensor(impl);
}

function _metaBinary(_keySet: unknown, self: Tensor, other: Tensor): Tensor {
  const shape = broadcastShapes(self.shape, other.shape);
  if (!shape) throw new Error(`Incompatible shapes: [${self.shape}] vs [${other.shape}]`);
  const dtype = resultDtype(self.dtype, other.dtype);
  return _metaTensor(shape, dtype);
}

function _metaUnary(_keySet: unknown, self: Tensor): Tensor {
  return _metaTensor([...self.shape], self.dtype);
}

function _metaReduction(_keySet: unknown, self: Tensor, dim?: number | readonly number[] | null, keepdim?: boolean): Tensor {
  const shape = [...self.shape];
  const dims = dim !== undefined && dim !== null
    ? (Array.isArray(dim) ? dim : [dim])
    : Array.from({ length: shape.length }, (_, i) => i);

  if (dims.length === shape.length || dim === undefined) {
    return _metaTensor(keepdim ? shape.map(() => 1) : [], self.dtype);
  }

  const result: number[] = [];
  const dimSet = new Set(dims.map(d => d < 0 ? shape.length + d : d));
  for (let i = 0; i < shape.length; i++) {
    if (dimSet.has(i)) {
      if (keepdim) result.push(1);
    } else {
      result.push(shape[i]);
    }
  }
  return _metaTensor(result, self.dtype);
}

export function metaMatmul(_keySet: unknown, self: Tensor, other: Tensor): Tensor {
  const shape = matmulOutputShape(self.shape, other.shape);
  if (shape === null) throw new Error(`metaMatmul: unsupported shapes`);
  return _metaTensor(shape, self.dtype);
}

export function metaClone(_keySet: unknown, self: Tensor): Tensor {
  return _metaTensor([...self.shape], self.dtype);
}

function mapKernel(names: readonly string[], kernel: NativeKernel): Record<string, NativeKernel> {
  return Object.fromEntries(names.map(name => [name, kernel]));
}

export const META_KERNELS: Readonly<Record<string, NativeKernel>> = Object.freeze({
  ...mapKernel(['add', 'sub', 'mul', 'div', 'pow', 'rem', 'maximum', 'minimum'], _metaBinary as NativeKernel),
  ...mapKernel([
    'neg', 'exp', 'log', 'sqrt', 'rsqrt', 'abs', 'sin', 'cos', 'tanh',
    'erf', 'erfc', 'lgamma', 'gamma', 'sigmoid', 'relu', 'gelu', 'silu',
    'sign', 'floor', 'ceil',
  ], _metaUnary as NativeKernel),
  ...mapKernel(['sum', 'mean', 'max', 'min', 'prod'], _metaReduction as NativeKernel),
  matmul: metaMatmul as NativeKernel,
  clone: metaClone as NativeKernel,
});
