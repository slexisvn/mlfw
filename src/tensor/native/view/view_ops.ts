import { Tensor } from '../../core/tensor.js';
import { TensorImpl } from '../../core/tensor_impl.js';
import { Storage } from '../../core/storage.js';
import { dtypeSize } from '../../types/dtype.js';
import type { NumericTypedArray } from '../../types/dtype.js';
import type { NativeKernel } from '../types.js';
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

type GpuContiguousHook = (tensor: Tensor) => Tensor | null;

function makeView(src: Tensor, newSizes: readonly number[], newStrides: readonly number[], offsetDelta?: number): Tensor {
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

export function reshapeKernel(tensor: Tensor, newShape: readonly number[]): Tensor {
  const result = inferReshape(tensor.shape, tensor.strides, newShape);
  if (!result) throw new Error(`Cannot reshape tensor of shape [${tensor.shape}] to [${newShape}]`);
  if (!result.needsCopy) return makeView(tensor, result.sizes, result.strides, 0);
  const contiguous = tensor.isContiguous ? tensor : contiguousKernel(tensor);
  return makeView(contiguous, result.sizes, result.strides, 0);
}

export function transposeKernel(tensor: Tensor, dim0: number, dim1: number): Tensor {
  const { sizes, strides } = computeTranspose(tensor.shape, tensor.strides, dim0, dim1);
  return makeView(tensor, sizes, strides, 0);
}

export function permuteKernel(tensor: Tensor, dims: readonly number[]): Tensor {
  const { sizes, strides } = computePermute(tensor.shape, tensor.strides, dims);
  return makeView(tensor, sizes, strides, 0);
}

export function expandKernel(tensor: Tensor, targetShape: readonly number[]): Tensor {
  const { sizes, strides } = computeExpand(tensor.shape, tensor.strides, targetShape);
  return makeView(tensor, sizes, strides, 0);
}

export function broadcastInDimKernel(tensor: Tensor, resultShape: readonly number[], broadcastDimensions: readonly number[]): Tensor {
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

export function sliceKernel(tensor: Tensor, dim: number, start: number, end: number | null, step: number): Tensor {
  const { sizes, strides, offsetDelta } = computeSlice(tensor.shape, tensor.strides, dim, start, end, step);
  return makeView(tensor, sizes, strides, offsetDelta);
}

export function unsqueezeKernel(tensor: Tensor, dim: number): Tensor {
  const { sizes, strides } = computeUnsqueeze(tensor.shape, tensor.strides, dim);
  return makeView(tensor, sizes, strides, 0);
}

export function squeezeKernel(tensor: Tensor, dim: number | null): Tensor {
  const { sizes, strides } = computeSqueeze(tensor.shape, tensor.strides, dim);
  return makeView(tensor, sizes, strides, 0);
}

export function narrowKernel(tensor: Tensor, dim: number, start: number, length: number): Tensor {
  const { sizes, strides, offsetDelta } = computeNarrow(tensor.shape, tensor.strides, dim, start, length);
  return makeView(tensor, sizes, strides, offsetDelta);
}

export function selectKernel(tensor: Tensor, dim: number, index: number): Tensor {
  const { sizes, strides, offsetDelta } = computeSelect(tensor.shape, tensor.strides, dim, index);
  return makeView(tensor, sizes, strides, offsetDelta);
}

let _gpuContiguousHook: GpuContiguousHook | null = null;
export function setGpuContiguousHook(fn: GpuContiguousHook | null): void { _gpuContiguousHook = fn; }

export function contiguousKernel(tensor: Tensor): Tensor {
  const impl = tensor._impl;
  if (tensor.isContiguous && impl.storageOffset === 0 && impl.storage.rawData!.length === tensor.numel) return tensor;
  return copyContiguous(tensor);
}

export const VIEW_KERNELS: Readonly<Record<string, NativeKernel>> = Object.freeze({
  reshape: (_keySet, self, shape) => reshapeKernel(self as Tensor, shape as readonly number[]),
  transpose: (_keySet, self, dim0, dim1) => transposeKernel(self as Tensor, dim0 as number, dim1 as number),
  permute: (_keySet, self, dims) => permuteKernel(self as Tensor, dims as readonly number[]),
  broadcast_in_dim: (_keySet, self, resultShape, broadcastDimensions) => broadcastInDimKernel(self as Tensor, resultShape as readonly number[], broadcastDimensions as readonly number[]),
  expand: (_keySet, self, shape) => expandKernel(self as Tensor, shape as readonly number[]),
  slice: (_keySet, self, dim, start, end, step) => sliceKernel(self as Tensor, dim as number, start as number, end as number | null, step as number),
  unsqueeze: (_keySet, self, dim) => unsqueezeKernel(self as Tensor, dim as number),
  squeeze: (_keySet, self, dim) => squeezeKernel(self as Tensor, dim as number | null),
  narrow: (_keySet, self, dim, start, length) => narrowKernel(self as Tensor, dim as number, start as number, length as number),
  select: (_keySet, self, dim, index) => selectKernel(self as Tensor, dim as number, index as number),
  contiguous: (_keySet, self) => contiguousKernel(self as Tensor),
});

function copyContiguous(tensor: Tensor): Tensor {
  if (_gpuContiguousHook) {
    const r = _gpuContiguousHook(tensor);
    if (r) return r;
  }

  const sizes = tensor.shape;
  const srcStrides = tensor.strides;
  const srcData = tensor._impl.storage.data as NumericTypedArray;
  const srcOffset = tensor._impl.storageOffset;
  const n = tensor.numel;
  const dstStrides = computeStrides(sizes);
  const dtype = tensor._impl.dtype;
  const newStorage = Storage.allocate(n * dtypeSize(dtype), dtype, tensor._impl.device);
  const dstData = newStorage.data as NumericTypedArray;
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
