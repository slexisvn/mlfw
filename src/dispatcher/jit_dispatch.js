import { DispatchKey } from './dispatch_key.js';
import { KernelFunction } from './boxing.js';
import { dispatcher } from './dispatcher.js';
import { jitCompile } from './jit_cache.js';
import { CPUTarget, GPUTarget, WasmTarget } from '../backend/target.js';
import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { computeStrides, computeNumel, broadcastShapes } from '../tensor/utils/shape_utils.js';
import { resultDtype, dtypeSize } from '../tensor/types/dtype.js';

const _TARGET_FOR_KEY = {
  [DispatchKey.CPU]: () => CPUTarget(),
  [DispatchKey.GPU]: () => GPUTarget(),
  [DispatchKey.WASM]: () => WasmTarget(),
};

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
    const aShape = tensorArgs[0].shape;
    const bShape = tensorArgs[1].shape;
    if (aShape.length === 1 && bShape.length === 1) return [];
    if (aShape.length === 2 && bShape.length === 2) return [aShape[0], bShape[1]];
    if (aShape.length === 2 && bShape.length === 1) return [aShape[0]];
    if (aShape.length >= 3) return [...aShape.slice(0, -2), aShape[aShape.length - 2], bShape[bShape.length - 1]];
    return [aShape[0], bShape[1]];
  }

  if (opName === 'dot') return [];

  if (opName === 'transpose') {
    const shape = [...tensorArgs[0].shape];
    const d0 = scalars.dim0 ?? 0;
    const d1 = scalars.dim1 ?? 1;
    const tmp = shape[d0]; shape[d0] = shape[d1]; shape[d1] = tmp;
    return shape;
  }

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

function _tensorToContiguous(t) {
  if (t.isContiguous) return t.data;
  const shape = t.shape;
  const strides = t.strides;
  const srcData = t._impl.storage.data;
  const srcOff = t._impl.storageOffset;
  const n = t.numel;
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

function _wrapResult(data, shape, dtype, device) {
  const strides = computeStrides(shape);
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, device);
  return new Tensor(impl);
}

function _wrapOpForJIT(opName, dispatchKey) {
  const getTarget = _TARGET_FOR_KEY[dispatchKey];
  if (!getTarget) return null;

  return (keySet, ...args) => {
    const { tensors, scalars } = _extractTensorsAndScalars(opName, args);
    if (tensors.length === 0) {
      throw new Error(`JIT dispatch: no tensor args for op '${opName}'`);
    }

    const target = getTarget();
    const entry = jitCompile(opName, tensors, scalars, target);

    const runtimeArgs = tensors.map(t => _tensorToContiguous(t));

    const outShape = _inferOutputShape(opName, tensors, scalars);
    const outDtype = resultDtype(tensors[0].dtype, tensors.length > 1 ? tensors[1].dtype : tensors[0].dtype);
    const outNumel = computeNumel(outShape);
    const Ctor = runtimeArgs.length > 0 ? runtimeArgs[0].constructor : Float32Array;
    const outData = new Ctor(Math.max(outNumel, 1));
    runtimeArgs.push(outData);

    entry.runtime.run(entry.funcName, ...runtimeArgs);

    return _wrapResult(outData, outShape, outDtype, tensors[0].device);
  };
}

export function registerJITKernels() {
  const ops = dispatcher.listOps();
  const backendKeys = [DispatchKey.CPU, DispatchKey.GPU, DispatchKey.WASM];

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
