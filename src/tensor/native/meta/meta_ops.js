import { Tensor } from '../../core/tensor.js';
import { TensorImpl } from '../../core/tensor_impl.js';
import { Storage } from '../../core/storage.js';
import { META_DEVICE } from '../../types/device.js';
import { broadcastShapes, computeStrides, matmulOutputShape } from '../../utils/shape_utils.js';
import { resultDtype } from '../../types/dtype.js';

function _metaTensor(shape, dtype) {
  const strides = computeStrides(shape);
  const storage = Storage.allocate(0, dtype, META_DEVICE);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, META_DEVICE);
  return new Tensor(impl);
}

function _metaBinary(keySet, self, other) {
  const shape = broadcastShapes(self.shape, other.shape);
  if (!shape) throw new Error(`Incompatible shapes: [${self.shape}] vs [${other.shape}]`);
  const dtype = resultDtype(self.dtype, other.dtype);
  return _metaTensor(shape, dtype);
}

function _metaUnary(keySet, self) {
  return _metaTensor([...self.shape], self.dtype);
}

function _metaReduction(keySet, self, dim, keepdim) {
  const shape = [...self.shape];
  const dims = dim !== undefined && dim !== null
    ? (Array.isArray(dim) ? dim : [dim])
    : Array.from({ length: shape.length }, (_, i) => i);

  if (dims.length === shape.length || dim === undefined) {
    return _metaTensor(keepdim ? shape.map(() => 1) : [], self.dtype);
  }

  const result = [];
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

export const metaAdd = _metaBinary;
export const metaSub = _metaBinary;
export const metaMul = _metaBinary;
export const metaDiv = _metaBinary;
export const metaPow = _metaBinary;
export const metaRem = _metaBinary;
export const metaMaximum = _metaBinary;
export const metaMinimum = _metaBinary;

export const metaNeg = _metaUnary;
export const metaExp = _metaUnary;
export const metaLog = _metaUnary;
export const metaSqrt = _metaUnary;
export const metaRsqrt = _metaUnary;
export const metaAbs = _metaUnary;
export const metaSin = _metaUnary;
export const metaCos = _metaUnary;
export const metaTanh = _metaUnary;
export const metaSigmoid = _metaUnary;
export const metaRelu = _metaUnary;
export const metaGelu = _metaUnary;
export const metaSilu = _metaUnary;
export const metaSign = _metaUnary;
export const metaFloor = _metaUnary;
export const metaCeil = _metaUnary;

export const metaSum = _metaReduction;
export const metaMean = _metaReduction;
export const metaMax = _metaReduction;
export const metaMin = _metaReduction;
export const metaProd = _metaReduction;

export function metaMatmul(keySet, self, other) {
  const shape = matmulOutputShape(self.shape, other.shape);
  if (shape === null) throw new Error(`metaMatmul: unsupported shapes`);
  return _metaTensor(shape, self.dtype);
}

export function metaClone(keySet, self) {
  return _metaTensor([...self.shape], self.dtype);
}
