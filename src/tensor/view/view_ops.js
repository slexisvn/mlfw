import { Tensor } from '../core/tensor.js';
import { TensorImpl } from '../core/tensor_impl.js';
import { inferReshape, computeStrides } from '../utils/shape_utils.js';
import {
  computeTranspose,
  computePermute,
  computeExpand,
  computeSlice,
  computeUnsqueeze,
  computeSqueeze,
  computeNarrow,
  computeSelect,
} from './view_utils.js';

function _makeView(src, newSizes, newStrides, offsetDelta) {
  const impl = new TensorImpl(
    src._impl.storage,
    src._impl.storageOffset + (offsetDelta || 0),
    newSizes,
    newStrides,
    src._impl.dtype,
    src._impl.device
  );
  return new Tensor(impl);
}

export function reshape(tensor, newShape) {
  const result = inferReshape(tensor.shape, tensor.strides, newShape);
  if (!result) throw new Error(`Cannot reshape tensor of shape [${tensor.shape}] to [${newShape}]`);

  if (!result.needsCopy) {
    return _makeView(tensor, result.sizes, result.strides, 0);
  }

  const contiguous = tensor.isContiguous ? tensor : _copyContiguous(tensor);
  return _makeView(contiguous, result.sizes, result.strides, 0);
}

export function transpose(tensor, dim0, dim1) {
  const { sizes, strides } = computeTranspose(tensor.shape, tensor.strides, dim0, dim1);
  return _makeView(tensor, sizes, strides, 0);
}

export function permute(tensor, dims) {
  const { sizes, strides } = computePermute(tensor.shape, tensor.strides, dims);
  return _makeView(tensor, sizes, strides, 0);
}

export function expand(tensor, targetShape) {
  const { sizes, strides } = computeExpand(tensor.shape, tensor.strides, targetShape);
  return _makeView(tensor, sizes, strides, 0);
}

export function slice(tensor, dim, start, end, step) {
  const { sizes, strides, offsetDelta } = computeSlice(
    tensor.shape, tensor.strides, dim, start, end, step
  );
  return _makeView(tensor, sizes, strides, offsetDelta);
}

export function unsqueeze(tensor, dim) {
  const { sizes, strides } = computeUnsqueeze(tensor.shape, tensor.strides, dim);
  return _makeView(tensor, sizes, strides, 0);
}

export function squeeze(tensor, dim) {
  const { sizes, strides } = computeSqueeze(tensor.shape, tensor.strides, dim);
  return _makeView(tensor, sizes, strides, 0);
}

export function narrow(tensor, dim, start, length) {
  const { sizes, strides, offsetDelta } = computeNarrow(
    tensor.shape, tensor.strides, dim, start, length
  );
  return _makeView(tensor, sizes, strides, offsetDelta);
}

export function select(tensor, dim, index) {
  const { sizes, strides, offsetDelta } = computeSelect(
    tensor.shape, tensor.strides, dim, index
  );
  return _makeView(tensor, sizes, strides, offsetDelta);
}

export function contiguous(tensor) {
  if (tensor.isContiguous) return tensor;
  return _copyContiguous(tensor);
}

function _copyContiguous(tensor) {
  const sizes = tensor.shape;
  const srcStrides = tensor.strides;
  const srcData = tensor._impl.storage.data;
  const srcOffset = tensor._impl.storageOffset;
  const n = tensor.numel;

  const dstStrides = computeStrides(sizes);

  const newStorage = tensor._impl.storage.clone();
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

  const impl = new TensorImpl(
    newStorage,
    0,
    sizes,
    dstStrides,
    tensor._impl.dtype,
    tensor._impl.device
  );
  return new Tensor(impl);
}

export function installViewOps(TensorClass) {
  const proto = TensorClass.prototype;
  proto.reshape = function(...args) {
    const shape = args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
    return reshape(this, shape);
  };
  proto.transpose = function(d0, d1) { return transpose(this, d0, d1); };
  proto.permute = function(...dims) {
    const d = dims.length === 1 && Array.isArray(dims[0]) ? dims[0] : dims;
    return permute(this, d);
  };
  proto.expand = function(...shape) {
    const s = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    return expand(this, s);
  };
  proto.slice = function(dim, start, end, step) { return slice(this, dim, start, end, step); };
  proto.unsqueeze = function(dim) { return unsqueeze(this, dim); };
  proto.squeeze = function(dim) { return squeeze(this, dim); };
  proto.narrow = function(dim, start, length) { return narrow(this, dim, start, length); };
  proto.select = function(dim, index) { return select(this, dim, index); };
  proto.contiguous = function() { return contiguous(this); };
  proto.t = function() {
    if (this.ndim !== 2) throw new Error('t() expects a 2D tensor');
    return transpose(this, 0, 1);
  };
}
