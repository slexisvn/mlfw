import { DispatchKey } from './dispatch_key.js';
import { KernelFunction } from './boxing.js';
import { dispatcher } from './dispatcher.js';
import { jitCompile } from './jit_cache.js';
import { isEagerDeferred } from './eager_mode.js';
import { CPUTarget, CUDATarget, WasmTarget, WebGPUTarget } from '../backend/target.js';
import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { computeStrides, computeNumel, broadcastShapes, matmulOutputShape } from '../tensor/utils/shape_utils.js';
import { resultDtype, typedArrayCtor } from '../tensor/types/dtype.js';

let _cpuTarget, _cudaTarget, _wasmTarget, _webgpuTarget;
const _TARGET_FOR_KEY = {
  [DispatchKey.CPU]: () => (_cpuTarget ??= CPUTarget()),
  [DispatchKey.GPU]: () => (_cudaTarget ??= CUDATarget()),
  [DispatchKey.WASM]: () => (_wasmTarget ??= WasmTarget()),
  [DispatchKey.CUSTOM_0]: () => (_webgpuTarget ??= WebGPUTarget()),
};

let _webgpuEagerFn = null;
export function setWebGPUEagerFn(fn) { _webgpuEagerFn = fn; }

const _SCALAR_ARG_SPEC = {
  sum: ['dim', 'keepdim'],
  mean: ['dim', 'keepdim'],
  max: ['dim', 'keepdim'],
  min: ['dim', 'keepdim'],
  prod: ['dim', 'keepdim'],
  argmax: ['dim', 'keepdim'],
  argmin: ['dim', 'keepdim'],
  transpose: ['dim0', 'dim1'],
  softmax: ['dim'],
  log_softmax: ['dim'],
  layer_norm: ['axis', 'eps'],
  batch_norm: ['axis', 'eps'],
  conv2d: ['strides', 'padding', 'dilation', 'groups'],
  pool2d: ['pool_type', 'kernel_size', 'strides', 'padding'],
  pad: ['low', 'high'],
  one_hot: ['depth'],
  index_select: ['dim'],
  gather: ['dim'],
  scatter_add: ['dim'],
  cat: ['dim'],
  stack: ['dim'],
};

function _extractTensorsAndScalars(opName, args) {
  const tensors = [];
  const scalars = {};
  const spec = _SCALAR_ARG_SPEC[opName];
  let scalarIdx = 0;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a && a._impl) {
      tensors.push(a);
    } else if (Array.isArray(a) && a.length > 0 && a[0] && a[0]._impl) {
      for (const el of a) {
        if (el && el._impl) tensors.push(el);
      }
    } else if (a !== undefined && a !== null) {
      if (spec && scalarIdx < spec.length) {
        scalars[spec[scalarIdx]] = a;
      }
      scalarIdx++;
    }
  }

  return { tensors, scalars };
}

function _inferOutputShape(opName, tensorArgs, scalars) {
  if (tensorArgs.length === 0) return [];

  const _REDUCE_SET = new Set(['sum', 'mean', 'max', 'min', 'prod', 'argmax', 'argmin']);
  if (_REDUCE_SET.has(opName)) {
    const inputShape = tensorArgs[0].shape;
    const dim = scalars.dim;
    const keepdim = scalars.keepdim;

    if (dim === undefined || dim === null) {
      return keepdim ? inputShape.map(() => 1) : [];
    }

    const dims = new Set(
      (Array.isArray(dim) ? dim : [dim]).map(d => d < 0 ? inputShape.length + d : d)
    );
    const result = [];
    for (let i = 0; i < inputShape.length; i++) {
      if (dims.has(i)) {
        if (keepdim) result.push(1);
      } else {
        result.push(inputShape[i]);
      }
    }
    return result;
  }

  if (opName === 'matmul') {
    return matmulOutputShape(tensorArgs[0].shape, tensorArgs[1].shape);
  }

  if (opName === 'dot') return [];

  if (opName === 'transpose') {
    const shape = [...tensorArgs[0].shape];
    const d0 = scalars.dim0 ?? 0;
    const d1 = scalars.dim1 ?? 1;
    const tmp = shape[d0]; shape[d0] = shape[d1]; shape[d1] = tmp;
    return shape;
  }

  if (opName === 'conv2d') {
    const inp = tensorArgs[0].shape;
    const w = tensorArgs[1].shape;
    const strides = scalars.strides || [1, 1];
    const padding = scalars.padding || [[0, 0], [0, 0]];
    const dilation = scalars.dilation || [1, 1];
    const spatial = [];
    for (let i = 0; i < 2; i++) {
      const padTotal = padding[i][0] + padding[i][1];
      const effK = (w[i + 2] - 1) * dilation[i] + 1;
      spatial.push(Math.floor((inp[i + 2] + padTotal - effK) / strides[i]) + 1);
    }
    return [inp[0], w[0], ...spatial];
  }

  if (opName === 'pool2d') {
    const inp = tensorArgs[0].shape;
    const ks = scalars.kernel_size || [2, 2];
    const strides = scalars.strides || ks;
    const padding = scalars.padding || [[0, 0], [0, 0]];
    const spatial = [];
    for (let i = 0; i < 2; i++) {
      const padTotal = padding[i][0] + padding[i][1];
      spatial.push(Math.floor((inp[i + 2] + padTotal - ks[i]) / strides[i]) + 1);
    }
    return [inp[0], inp[1], ...spatial];
  }

  if (opName === 'clamp') {
    let shape = tensorArgs[0].shape;
    for (let i = 1; i < tensorArgs.length; i++) {
      shape = broadcastShapes(shape, tensorArgs[i].shape) || shape;
    }
    return shape;
  }

  if (opName === 'pad') {
    const inp = tensorArgs[0].shape;
    const low = scalars.low || [];
    const high = scalars.high || [];
    return inp.map((d, i) => d + (low[i] || 0) + (high[i] || 0));
  }

  if (opName === 'one_hot') {
    return [...tensorArgs[0].shape, scalars.depth];
  }

  if (opName === 'cat') {
    const rank = tensorArgs[0].shape.length;
    const dim = (scalars.dim ?? 0) < 0 ? rank + (scalars.dim ?? 0) : (scalars.dim ?? 0);
    const shape = [...tensorArgs[0].shape];
    shape[dim] = tensorArgs.reduce((acc, t) => acc + t.shape[dim], 0);
    return shape;
  }

  if (opName === 'stack') {
    const rank = tensorArgs[0].shape.length;
    const dim = (scalars.dim ?? 0) < 0 ? rank + 1 + (scalars.dim ?? 0) : (scalars.dim ?? 0);
    const shape = [...tensorArgs[0].shape];
    shape.splice(dim, 0, tensorArgs.length);
    return shape;
  }

  if (opName === 'index_select') {
    const shape = [...tensorArgs[0].shape];
    const rank = shape.length;
    const dim = (scalars.dim ?? 0) < 0 ? rank + (scalars.dim ?? 0) : (scalars.dim ?? 0);
    shape[dim] = tensorArgs[1].shape.reduce((a, b) => a * b, 1);
    return shape;
  }

  if (opName === 'gather') return [...tensorArgs[1].shape];
  if (opName === 'scatter_add') return [...tensorArgs[0].shape];

  if (opName === 'softmax' || opName === 'log_softmax') return [...tensorArgs[0].shape];
  if (opName === 'layer_norm' || opName === 'batch_norm') return [...tensorArgs[0].shape];
  if (opName === 'embedding') {
    const weightShape = tensorArgs[0].shape;
    const indicesShape = tensorArgs[1].shape;
    return [...indicesShape, weightShape[1]];
  }

  if (tensorArgs.length === 1) return [...tensorArgs[0].shape];

  const bcast = broadcastShapes(tensorArgs[0].shape, tensorArgs[1].shape);
  return bcast || [...tensorArgs[0].shape];
}

export function tensorToContiguous(t) {
  const srcData = t._impl.storage.data;
  const srcOff = t._impl.storageOffset;
  const n = t.numel;
  if (t.isContiguous && srcOff === 0 && srcData.length === n) return t.data;
  const shape = t.shape;
  const strides = t.strides;
  const Ctor = srcData.constructor;
  const dst = new Ctor(n);
  const ndim = shape.length;
  const indices = new Int32Array(ndim);
  let si = srcOff;
  for (let i = 0; i < n; i++) {
    dst[i] = srcData[si];
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < shape[d]) { si += strides[d]; break; }
      si -= (shape[d] - 1) * strides[d];
      indices[d] = 0;
    }
  }
  return dst;
}

export function wrapResult(data, shape, dtype, device) {
  const strides = computeStrides(shape);
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, device);
  return new Tensor(impl);
}

function _prod(arr, from, to) {
  let p = 1;
  for (let i = from; i < to; i++) p *= arr[i];
  return p;
}

function _hostStack(inputs, inShape, dim, outData) {
  const rank = inShape.length;
  const d = dim < 0 ? rank + 1 + dim : dim;
  const outer = _prod(inShape, 0, d);
  const inner = _prod(inShape, d, rank);
  const n = inputs.length;
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < n; i++) {
      outData.set(inputs[i].subarray(o * inner, (o + 1) * inner), (o * n + i) * inner);
    }
  }
}

function _hostCat(inputs, shapes, dim, outData) {
  const rank = shapes[0].length;
  const d = dim < 0 ? rank + dim : dim;
  const outer = _prod(shapes[0], 0, d);
  const tail = _prod(shapes[0], d + 1, rank);
  let outDimTotal = 0;
  for (const s of shapes) outDimTotal += s[d];
  for (let o = 0; o < outer; o++) {
    let outOff = o * outDimTotal * tail;
    for (let k = 0; k < inputs.length; k++) {
      const block = shapes[k][d] * tail;
      outData.set(inputs[k].subarray(o * block, (o + 1) * block), outOff);
      outOff += block;
    }
  }
}

function _runHostConcatLike(opName, tensors, scalars) {
  const outShape = _inferOutputShape(opName, tensors, scalars);
  const outDtype = tensors[0].dtype;
  const Ctor = typedArrayCtor(outDtype);
  const outData = new Ctor(Math.max(computeNumel(outShape), 1));
  const dim = scalars.dim ?? 0;
  if (isEagerDeferred() && _gpuConcatFn) {
    const inputArrays = tensors.map(t => _gpuInputArray(t));
    _gpuConcatFn(opName, inputArrays, tensors.map(t => t.shape), dim, outShape, outData, outDtype);
    return wrapResult(outData, outShape, outDtype, tensors[0].device);
  }
  const runtimeArgs = tensors.map(t => tensorToContiguous(t));
  if (opName === 'stack') _hostStack(runtimeArgs, tensors[0].shape, dim, outData);
  else _hostCat(runtimeArgs, tensors.map(t => t.shape), dim, outData);
  return wrapResult(outData, outShape, outDtype, tensors[0].device);
}

let _gpuContiguousFn = null;
export function setGpuContiguousFn(fn) { _gpuContiguousFn = fn; }
export function getGpuContiguousFn() { return _gpuContiguousFn; }
let _gpuConcatFn = null;
export function setGpuConcatFn(fn) { _gpuConcatFn = fn; }
let _cudnnLSTM = null;
export function setCudnnLSTM(fn) { _cudnnLSTM = fn; }
export function getCudnnLSTM() { return _cudnnLSTM; }
let _webgpuRNN = null;
export function setWebgpuRNN(fn) { _webgpuRNN = fn; }
export function getWebgpuRNN() { return _webgpuRNN; }
let _gpuAdam = null;
export function setGpuAdamFn(fn) { _gpuAdam = fn; }
export function getGpuAdamFn() { return _gpuAdam; }
let _gpuMatmul = null;
export function setGpuMatmul(fn) { _gpuMatmul = fn; }
export function getGpuMatmul() { return _gpuMatmul; }

function _gpuInputArray(t) {
  if (t.isContiguous && t._impl.storageOffset === 0) {
    const raw = t._impl.storage.rawData;
    if (raw && raw.length === t.numel) return raw;
  }
  if (isEagerDeferred() && _gpuContiguousFn) {
    return _gpuContiguousFn(t._impl.storage.rawData, t.shape, t.strides, t._impl.storageOffset, t.dtype);
  }
  return tensorToContiguous(t);
}
export function gpuContiguousArray(t) { return _gpuInputArray(t); }

function _wrapOpForJIT(opName, dispatchKey) {
  const getTarget = _TARGET_FOR_KEY[dispatchKey];
  if (!getTarget) return null;

  const isGPU = dispatchKey === DispatchKey.GPU;
  const isWebGPU = dispatchKey === DispatchKey.CUSTOM_0;
  const hostConcatLike = isGPU && (opName === 'stack' || opName === 'cat');

  return (keySet, ...args) => {
    const { tensors, scalars } = _extractTensorsAndScalars(opName, args);
    if (tensors.length === 0) {
      throw new Error(`JIT dispatch: no tensor args for op '${opName}'`);
    }

    if (hostConcatLike) return _runHostConcatLike(opName, tensors, scalars);

    const target = getTarget();
    const entry = jitCompile(opName, tensors, scalars, target);

    const outShape = _inferOutputShape(opName, tensors, scalars);
    const outDtype = entry.outDtype || resultDtype(tensors[0].dtype, tensors.length > 1 ? tensors[1].dtype : tensors[0].dtype);
    const outNumel = computeNumel(outShape);
    const Ctor = typedArrayCtor(outDtype);
    const outData = new Ctor(Math.max(outNumel, 1));

    if (isWebGPU) {
      _webgpuEagerFn(entry.compiled, tensors, outData);
      return wrapResult(outData, outShape, outDtype, tensors[0].device);
    }

    const runtimeArgs = tensors.map(t => isGPU ? _gpuInputArray(t) : tensorToContiguous(t));
    runtimeArgs.push(outData);
    entry.runtime.run(entry.funcName, ...runtimeArgs);

    return wrapResult(outData, outShape, outDtype, tensors[0].device);
  };
}

export function registerJITKernels() {
  const ops = dispatcher.listOps();
  const backendKeys = [DispatchKey.CPU, DispatchKey.GPU, DispatchKey.WASM, DispatchKey.CUSTOM_0];

  for (const opKey of ops) {
    const handle = dispatcher.findOp(opKey);
    if (!handle) continue;
    const opName = handle.name;

    for (const key of backendKeys) {
      if (handle.entry.hasKernel(key)) continue;
      const kernel = _wrapOpForJIT(opName, key);
      if (kernel) {
        handle.entry.registerKernel(key, KernelFunction.fromUnboxed(kernel));
      }
    }
  }
}
