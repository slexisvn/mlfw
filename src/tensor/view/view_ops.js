import { Tensor } from '../core/tensor.js';
import { TensorImpl } from '../core/tensor_impl.js';
import { Storage } from '../core/storage.js';
import { dtypeSize } from '../types/dtype.js';
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
import { AutogradMeta } from '../core/autograd_meta.js';

let _GradMode = null;
let _ReshapeBackward = null;
let _TransposeBackward = null;
let _PermuteBackward = null;
let _SliceBackward = null;
let _ExpandBackward = null;
let _SelectBackward = null;
let _GradAccumulator = null;

export function _initViewAutograd(reg) {
  _GradMode = reg.GradMode;
  _ReshapeBackward = reg.ReshapeBackward;
  _TransposeBackward = reg.TransposeBackward;
  _PermuteBackward = reg.PermuteBackward;
  _SliceBackward = reg.SliceBackward;
  _ExpandBackward = reg.ExpandBackward;
  _SelectBackward = reg.SelectBackward;
  _GradAccumulator = reg.GradAccumulator;
}

function _wrapWithAutograd(srcTensor, resultTensor, BackwardClass, ...ctorArgs) {
  if (!_GradMode || !_GradMode.isEnabled()) return;
  const srcMeta = srcTensor._impl.autogradMeta;
  if (!srcMeta || !srcMeta.requiresGrad) return;

  const node = new BackwardClass(...ctorArgs);
  node.saveInputMetadata(0, [...srcTensor.shape], srcTensor.dtype);

  const srcFn = srcMeta.gradFn;
  if (srcFn) {
    node.setNextEdge(0, srcFn, srcMeta.outputNr || 0);
  } else {
    let acc = srcMeta.getGradAccumulator();
    if (!acc && _GradAccumulator) {
      acc = new _GradAccumulator(srcTensor);
      srcMeta.setGradAccumulator(acc);
    }
    if (acc) node.setNextEdge(0, acc, 0);
  }

  const meta = new AutogradMeta();
  meta.setGradFn(node, 0);
  meta.requiresGrad = true;
  resultTensor._impl.setAutogradMeta(meta);
  resultTensor._impl._updateKeySet();
}

function _traceView(tensor, opName, attrs) {
  return tensor.tracer.recordOp(opName, [tensor], attrs);
}

function _resolveShape(newShape, srcShape) {
  const total = srcShape.reduce((a, b) => a * b, 1);
  let known = 1, inferIdx = -1;
  const out = newShape.slice();
  for (let i = 0; i < out.length; i++) {
    if (out[i] === -1) inferIdx = i;
    else known *= out[i];
  }
  if (inferIdx >= 0) out[inferIdx] = known === 0 ? 0 : total / known;
  return out;
}

function _makeView(src, newSizes, newStrides, offsetDelta) {
  const impl = new TensorImpl(
    src._impl.storage,
    src._impl.storageOffset + (offsetDelta || 0),
    newSizes,
    newStrides,
    src._impl.dtype,
    src._impl.device
  );
  const srcMeta = src._impl.autogradMeta;
  if (srcMeta) {
    impl.setAutogradMeta(srcMeta);
  }
  return new Tensor(impl);
}

export function reshape(tensor, newShape) {
  if (tensor.isSymbolic && tensor.tracer) {
    return _traceView(tensor, 'reshape', { new_shape: _resolveShape(newShape, tensor.shape) });
  }
  const result = inferReshape(tensor.shape, tensor.strides, newShape);
  if (!result) throw new Error(`Cannot reshape tensor of shape [${tensor.shape}] to [${newShape}]`);

  let out;
  if (!result.needsCopy) {
    out = _makeView(tensor, result.sizes, result.strides, 0);
  } else {
    const contiguous = tensor.isContiguous ? tensor : _copyContiguous(tensor);
    out = _makeView(contiguous, result.sizes, result.strides, 0);
  }
  if (_ReshapeBackward) _wrapWithAutograd(tensor, out, _ReshapeBackward);
  return out;
}

export function transpose(tensor, dim0, dim1) {
  if (tensor.isSymbolic && tensor.tracer) {
    const rank = tensor.shape.length;
    const d0 = dim0 < 0 ? rank + dim0 : dim0;
    const d1 = dim1 < 0 ? rank + dim1 : dim1;
    return _traceView(tensor, 'transpose', { dim0: d0, dim1: d1 });
  }
  const { sizes, strides } = computeTranspose(tensor.shape, tensor.strides, dim0, dim1);
  const out = _makeView(tensor, sizes, strides, 0);
  if (_TransposeBackward) _wrapWithAutograd(tensor, out, _TransposeBackward, dim0, dim1);
  return out;
}

export function permute(tensor, dims) {
  if (tensor.isSymbolic && tensor.tracer) {
    const rank = tensor.shape.length;
    const resolved = dims.map(d => d < 0 ? rank + d : d);
    return _traceView(tensor, 'permute', { dims: resolved });
  }
  const { sizes, strides } = computePermute(tensor.shape, tensor.strides, dims);
  const out = _makeView(tensor, sizes, strides, 0);
  if (_PermuteBackward) _wrapWithAutograd(tensor, out, _PermuteBackward, dims);
  return out;
}

export function expand(tensor, targetShape) {
  if (tensor.isSymbolic && tensor.tracer) {
    const inRank = tensor.shape.length;
    const offset = targetShape.length - inRank;
    const resultShape = targetShape.map((d, i) => d === -1 ? tensor.shape[i - offset] : d);
    const broadcastDimensions = Array.from({ length: inRank }, (_, i) => i + offset);
    return _traceView(tensor, 'broadcast_in_dim', { result_shape: resultShape, broadcast_dimensions: broadcastDimensions });
  }
  const { sizes, strides } = computeExpand(tensor.shape, tensor.strides, targetShape);
  const out = _makeView(tensor, sizes, strides, 0);
  if (_ExpandBackward) _wrapWithAutograd(tensor, out, _ExpandBackward);
  return out;
}

function _sliceAttrs(shape, dim, start, end, step) {
  const rank = shape.length;
  const d = dim < 0 ? rank + dim : dim;
  const dimSize = shape[d];
  let s = start ?? 0;
  let e = end ?? dimSize;
  const st = step ?? 1;
  if (s < 0) s += dimSize;
  if (e < 0) e += dimSize;
  s = Math.max(0, Math.min(s, dimSize));
  e = Math.max(0, Math.min(e, dimSize));
  const starts = new Array(rank).fill(0);
  const limits = shape.slice();
  const strides = new Array(rank).fill(1);
  starts[d] = s;
  limits[d] = e;
  strides[d] = st;
  return { starts, limits, strides, d };
}

export function slice(tensor, dim, start, end, step) {
  if (tensor.isSymbolic && tensor.tracer) {
    const { starts, limits, strides } = _sliceAttrs(tensor.shape, dim, start, end, step);
    return _traceView(tensor, 'slice', { starts, limits, strides });
  }
  const { sizes, strides, offsetDelta } = computeSlice(
    tensor.shape, tensor.strides, dim, start, end, step
  );
  const out = _makeView(tensor, sizes, strides, offsetDelta);
  if (_SliceBackward) {
    const a = _sliceAttrs(tensor.shape, dim, start, end, step);
    _wrapWithAutograd(tensor, out, _SliceBackward, a.d, a.starts[a.d], a.limits[a.d], a.strides[a.d]);
  }
  return out;
}

export function unsqueeze(tensor, dim) {
  if (tensor.isSymbolic && tensor.tracer) {
    const rank = tensor.shape.length;
    const d = dim < 0 ? rank + 1 + dim : dim;
    const newShape = tensor.shape.slice();
    newShape.splice(d, 0, 1);
    return _traceView(tensor, 'reshape', { new_shape: newShape });
  }
  const { sizes, strides } = computeUnsqueeze(tensor.shape, tensor.strides, dim);
  const out = _makeView(tensor, sizes, strides, 0);
  if (_ReshapeBackward) _wrapWithAutograd(tensor, out, _ReshapeBackward);
  return out;
}

export function squeeze(tensor, dim) {
  if (tensor.isSymbolic && tensor.tracer) {
    const { sizes } = computeSqueeze(tensor.shape, tensor.strides, dim);
    return _traceView(tensor, 'reshape', { new_shape: sizes });
  }
  const { sizes, strides } = computeSqueeze(tensor.shape, tensor.strides, dim);
  const out = _makeView(tensor, sizes, strides, 0);
  if (_ReshapeBackward) _wrapWithAutograd(tensor, out, _ReshapeBackward);
  return out;
}

export function narrow(tensor, dim, start, length) {
  if (tensor.isSymbolic && tensor.tracer) {
    const { starts, limits, strides } = _sliceAttrs(tensor.shape, dim, start, start + length, 1);
    return _traceView(tensor, 'slice', { starts, limits, strides });
  }
  const { sizes, strides, offsetDelta } = computeNarrow(
    tensor.shape, tensor.strides, dim, start, length
  );
  const out = _makeView(tensor, sizes, strides, offsetDelta);
  if (_SliceBackward) {
    const a = _sliceAttrs(tensor.shape, dim, start, start + length, 1);
    _wrapWithAutograd(tensor, out, _SliceBackward, a.d, a.starts[a.d], a.limits[a.d], a.strides[a.d]);
  }
  return out;
}

export function select(tensor, dim, index) {
  if (tensor.isSymbolic && tensor.tracer) {
    const rank = tensor.shape.length;
    const d = dim < 0 ? rank + dim : dim;
    const idx = index < 0 ? tensor.shape[d] + index : index;
    const { starts, limits, strides } = _sliceAttrs(tensor.shape, d, idx, idx + 1, 1);
    const sliced = _traceView(tensor, 'slice', { starts, limits, strides });
    const newShape = tensor.shape.filter((_, i) => i !== d);
    return sliced.tracer.recordOp('reshape', [sliced], { new_shape: newShape });
  }
  const { sizes, strides, offsetDelta } = computeSelect(
    tensor.shape, tensor.strides, dim, index
  );
  const out = _makeView(tensor, sizes, strides, offsetDelta);
  if (_SelectBackward) {
    const rank = tensor.shape.length;
    const d = dim < 0 ? rank + dim : dim;
    const idx = index < 0 ? tensor.shape[d] + index : index;
    _wrapWithAutograd(tensor, out, _SelectBackward, d, idx);
  }
  return out;
}

let _gpuContiguousHook = null;
export function setGpuContiguousHook(fn) { _gpuContiguousHook = fn; }

export function contiguous(tensor) {
  const impl = tensor._impl;
  if (tensor.isContiguous && impl.storageOffset === 0 && impl.storage.rawData.length === tensor.numel) {
    return tensor;
  }
  return _copyContiguous(tensor);
}

function _copyContiguous(tensor) {
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

  const impl = new TensorImpl(
    newStorage,
    0,
    sizes,
    dstStrides,
    tensor._impl.dtype,
    tensor._impl.device
  );
  const srcMeta = tensor._impl.autogradMeta;
  if (srcMeta) {
    impl.setAutogradMeta(srcMeta);
  }
  return new Tensor(impl);
}

export function repeat(tensor, reps) {
  const shape = tensor.shape;
  const ndim = shape.length;
  if (reps.length < ndim) throw new Error('repeat: reps length must be >= tensor rank');
  const lead = reps.length - ndim;
  const aligned = lead > 0 ? [...Array(lead).fill(1), ...shape] : shape.slice();
  const interleaved = [];
  const expanded = [];
  const finalShape = [];
  for (let i = 0; i < aligned.length; i++) {
    interleaved.push(1, aligned[i]);
    expanded.push(reps[i], aligned[i]);
    finalShape.push(reps[i] * aligned[i]);
  }
  return reshape(expand(reshape(tensor, interleaved), expanded), finalShape);
}

export function tile(tensor, reps) {
  const ndim = tensor.shape.length;
  const r = reps.length < ndim ? [...Array(ndim - reps.length).fill(1), ...reps] : reps;
  return repeat(tensor, r);
}

export function split(tensor, sizeOrSizes, dim = 0) {
  const rank = tensor.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  const n = tensor.shape[d];
  let sizes;
  if (Array.isArray(sizeOrSizes)) {
    sizes = sizeOrSizes;
  } else {
    sizes = [];
    for (let off = 0; off < n; off += sizeOrSizes) sizes.push(Math.min(sizeOrSizes, n - off));
  }
  const out = [];
  let start = 0;
  for (const s of sizes) {
    out.push(narrow(tensor, d, start, s));
    start += s;
  }
  return out;
}

export function chunk(tensor, chunks, dim = 0) {
  const rank = tensor.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  const n = tensor.shape[d];
  const size = Math.ceil(n / chunks);
  return split(tensor, size, d);
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
  proto.repeat = function(...reps) {
    const r = reps.length === 1 && Array.isArray(reps[0]) ? reps[0] : reps;
    return repeat(this, r);
  };
  proto.tile = function(...reps) {
    const r = reps.length === 1 && Array.isArray(reps[0]) ? reps[0] : reps;
    return tile(this, r);
  };
  proto.split = function(sizeOrSizes, dim = 0) { return split(this, sizeOrSizes, dim); };
  proto.chunk = function(chunks, dim = 0) { return chunk(this, chunks, dim); };
  proto.contiguous = function() { return contiguous(this); };
  proto.t = function() {
    if (this.ndim !== 2) throw new Error('t() expects a 2D tensor');
    return transpose(this, 0, 1);
  };
}
