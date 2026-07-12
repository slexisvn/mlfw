import { Tensor } from '../../core/tensor.js';
import { TensorImpl } from '../../core/tensor_impl.js';
import { Storage } from '../../core/storage.js';
import { dtypeSize } from '../../types/dtype.js';
import { inferReshape, computeStrides } from '../../utils/shape_utils.js';
import {
  computeTranspose,
  computePermute,
  computeExpand,
  computeSlice,
  computeUnsqueeze,
  computeSqueeze,
  computeNarrow,
  computeSelect,
} from '../../utils/view_utils.js';

function makeView(src, newSizes, newStrides, offsetDelta) {
  const impl = new TensorImpl(
    src._impl.storage,
    src._impl.storageOffset + (offsetDelta || 0),
    newSizes,
    newStrides,
    src._impl.dtype,
    src._impl.device
  );
  const srcMeta = src._impl.autogradMeta;
  if (srcMeta) impl.setAutogradMeta(srcMeta);
  return new Tensor(impl);
}

export function reshapeKernel(tensor, newShape) {
  const result = inferReshape(tensor.shape, tensor.strides, newShape);
  if (!result) throw new Error(`Cannot reshape tensor of shape [${tensor.shape}] to [${newShape}]`);
  if (!result.needsCopy) return makeView(tensor, result.sizes, result.strides, 0);
  const contiguous = tensor.isContiguous ? tensor : contiguousKernel(tensor);
  return makeView(contiguous, result.sizes, result.strides, 0);
}

export function transposeKernel(tensor, dim0, dim1) {
  const { sizes, strides } = computeTranspose(tensor.shape, tensor.strides, dim0, dim1);
  return makeView(tensor, sizes, strides, 0);
}

export function permuteKernel(tensor, dims) {
  const { sizes, strides } = computePermute(tensor.shape, tensor.strides, dims);
  return makeView(tensor, sizes, strides, 0);
}

export function expandKernel(tensor, targetShape) {
  const { sizes, strides } = computeExpand(tensor.shape, tensor.strides, targetShape);
  return makeView(tensor, sizes, strides, 0);
}

export function broadcastInDimKernel(tensor, resultShape, broadcastDimensions) {
  const srcShape = tensor.shape;
  const srcStrides = tensor.strides;
  const sizes = [...resultShape];
  const strides = new Array(resultShape.length).fill(0);
  for (let i = 0; i < broadcastDimensions.length; i++) {
    const outDim = broadcastDimensions[i];
    strides[outDim] = srcShape[i] === 1 && resultShape[outDim] !== 1 ? 0 : srcStrides[i];
  }
  return makeView(tensor, sizes, strides, 0);
}

export function sliceKernel(tensor, dim, start, end, step) {
  const { sizes, strides, offsetDelta } = computeSlice(tensor.shape, tensor.strides, dim, start, end, step);
  return makeView(tensor, sizes, strides, offsetDelta);
}

export function unsqueezeKernel(tensor, dim) {
  const { sizes, strides } = computeUnsqueeze(tensor.shape, tensor.strides, dim);
  return makeView(tensor, sizes, strides, 0);
}

export function squeezeKernel(tensor, dim) {
  const { sizes, strides } = computeSqueeze(tensor.shape, tensor.strides, dim);
  return makeView(tensor, sizes, strides, 0);
}

export function narrowKernel(tensor, dim, start, length) {
  const { sizes, strides, offsetDelta } = computeNarrow(tensor.shape, tensor.strides, dim, start, length);
  return makeView(tensor, sizes, strides, offsetDelta);
}

export function selectKernel(tensor, dim, index) {
  const { sizes, strides, offsetDelta } = computeSelect(tensor.shape, tensor.strides, dim, index);
  return makeView(tensor, sizes, strides, offsetDelta);
}

let _gpuContiguousHook = null;
export function setGpuContiguousHook(fn) { _gpuContiguousHook = fn; }

export function contiguousKernel(tensor) {
  const impl = tensor._impl;
  if (tensor.isContiguous && impl.storageOffset === 0 && impl.storage.rawData.length === tensor.numel) return tensor;
  return copyContiguous(tensor);
}

export const VIEW_KERNELS = Object.freeze({
  reshape: (keySet, self, shape) => reshapeKernel(self, shape),
  transpose: (keySet, self, dim0, dim1) => transposeKernel(self, dim0, dim1),
  permute: (keySet, self, dims) => permuteKernel(self, dims),
  broadcast_in_dim: (keySet, self, resultShape, broadcastDimensions) => broadcastInDimKernel(self, resultShape, broadcastDimensions),
  expand: (keySet, self, shape) => expandKernel(self, shape),
  slice: (keySet, self, dim, start, end, step) => sliceKernel(self, dim, start, end, step),
  unsqueeze: (keySet, self, dim) => unsqueezeKernel(self, dim),
  squeeze: (keySet, self, dim) => squeezeKernel(self, dim),
  narrow: (keySet, self, dim, start, length) => narrowKernel(self, dim, start, length),
  select: (keySet, self, dim, index) => selectKernel(self, dim, index),
  contiguous: (keySet, self) => contiguousKernel(self),
});

function copyContiguous(tensor) {
  if (_gpuContiguousHook) {
    const r = _gpuContiguousHook(tensor);
    if (r) return r;
  }

  const sizes = tensor.shape;
  const srcStrides = tensor.strides;
  const srcData = tensor._impl.storage.data;
  const srcOffset = tensor._impl.storageOffset;
  const n = tensor.numel;
  const dstStrides = computeStrides(sizes);
  const dtype = tensor._impl.dtype;
  const newStorage = Storage.allocate(n * dtypeSize(dtype), dtype, tensor._impl.device);
  const dstData = newStorage.data;
  const ndim = sizes.length;
  const indices = new Int32Array(ndim);
  let srcIdx = srcOffset;

  for (let i = 0; i < n; i++) {
    dstData[i] = srcData[srcIdx];
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      srcIdx += srcStrides[d];
      if (indices[d] < sizes[d]) break;
      srcIdx -= indices[d] * srcStrides[d];
      indices[d] = 0;
    }
  }

  const impl = new TensorImpl(newStorage, 0, sizes, dstStrides, tensor._impl.dtype, tensor._impl.device);
  const srcMeta = tensor._impl.autogradMeta;
  if (srcMeta) impl.setAutogradMeta(srcMeta);
  return new Tensor(impl);
}
