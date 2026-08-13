import { DispatchKey } from './dispatch_key.js';
import type { DispatchKeyValue, DispatchKeySet } from './dispatch_key.js';
import { KernelFunction } from './boxing.js';
import { dispatcher } from './dispatcher.js';
import { jitCompile } from './jit_cache.js';
import type { TargetLike } from './jit_cache.js';
import { isEagerDeferred } from './eager_mode.js';
import { CPUTarget, CUDATarget, WasmTarget, WebGPUTarget } from '../backend/target.js';
import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { computeStrides, computeNumel, broadcastShapes, matmulOutputShape } from '../tensor/utils/shape_utils.js';
import { resultDtype, typedArrayCtor } from '../tensor/types/dtype.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { Device } from '../tensor/types/device.js';
import type { MutableNumericArray, NumericSettable } from '../tensor/types/options.js';
import { scalarArgNames } from '../tensor/ops/metadata.js';

type ScalarMap = Record<string, unknown>;
type TargetFactory = () => TargetLike;
type GpuContiguousFn = (data: NumericTypedArray | null, shape: readonly number[], strides: readonly number[], offset: number, dtype: DType) => NumericTypedArray;
type GpuConcatFn = (opName: string, inputs: NumericTypedArray[], shapes: readonly number[][], dim: number, outShape: readonly number[], outData: NumericTypedArray, dtype: DType) => void;
export type DynamicFn = (...args: unknown[]) => unknown;

let _cpuTarget: TargetLike | undefined;
let _cudaTarget: TargetLike | undefined;
let _wasmTarget: TargetLike | undefined;
let _webgpuTarget: TargetLike | undefined;
const _TARGET_FOR_KEY: Partial<Record<DispatchKeyValue, TargetFactory>> = {
  [DispatchKey.CPU]: () => (_cpuTarget ??= CPUTarget()),
  [DispatchKey.GPU]: () => (_cudaTarget ??= CUDATarget()),
  [DispatchKey.WASM]: () => (_wasmTarget ??= WasmTarget()),
  [DispatchKey.CUSTOM_0]: () => (_webgpuTarget ??= WebGPUTarget()),
};

let _webgpuEagerFn: DynamicFn | null = null;
export function setWebGPUEagerFn(fn: DynamicFn | null) { _webgpuEagerFn = fn; }

function hasTensorImpl(value: unknown): value is Tensor {
  return typeof value === 'object' && value !== null && '_impl' in value;
}

function _extractTensorsAndScalars(opName: string, args: readonly unknown[]): { tensors: Tensor[]; scalars: ScalarMap } {
  const tensors: Tensor[] = [];
  const scalars: ScalarMap = {};
  const spec = scalarArgNames(opName);
  let scalarIdx = 0;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (hasTensorImpl(a)) {
      tensors.push(a);
    } else if (Array.isArray(a) && a.length > 0 && hasTensorImpl(a[0])) {
      for (const el of a) {
        if (hasTensorImpl(el)) tensors.push(el);
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

function numberScalar(value: unknown, fallback: number): number {
  return typeof value === 'number' ? value : fallback;
}

function numberArrayScalar(value: unknown, fallback: readonly number[]): readonly number[] {
  return Array.isArray(value) ? value as readonly number[] : fallback;
}

function paddingScalar(value: unknown): readonly (readonly number[])[] {
  return Array.isArray(value) ? value as readonly (readonly number[])[] : [[0, 0], [0, 0]];
}

function _inferOutputShape(opName: string, tensorArgs: readonly Tensor[], scalars: ScalarMap): number[] {
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
      (Array.isArray(dim) ? dim : [dim]).map(d => typeof d === 'number' && d < 0 ? inputShape.length + d : Number(d))
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
    const shape = matmulOutputShape(tensorArgs[0].shape, tensorArgs[1].shape);
    if (shape === null) {
      throw new Error(`matmul: incompatible shapes [${tensorArgs[0].shape}] and [${tensorArgs[1].shape}] — the last dim of the first operand must equal the second-to-last dim of the second`);
    }
    return shape;
  }

  if (opName === 'dot') return [];

  if (opName === 'transpose') {
    const shape = [...tensorArgs[0].shape];
    const d0 = numberScalar(scalars.dim0, 0);
    const d1 = numberScalar(scalars.dim1, 1);
    const tmp = shape[d0]; shape[d0] = shape[d1]; shape[d1] = tmp;
    return shape;
  }

  if (opName === 'conv2d') {
    const inp = tensorArgs[0].shape;
    const w = tensorArgs[1].shape;
    const strides = numberArrayScalar(scalars.strides, [1, 1]);
    const padding = paddingScalar(scalars.padding);
    const dilation = numberArrayScalar(scalars.dilation, [1, 1]);
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
    const ks = numberArrayScalar(scalars.kernel_size, [2, 2]);
    const strides = numberArrayScalar(scalars.strides, ks);
    const padding = paddingScalar(scalars.padding);
    const spatial = [];
    for (let i = 0; i < 2; i++) {
      const padTotal = padding[i][0] + padding[i][1];
      spatial.push(Math.floor((inp[i + 2] + padTotal - ks[i]) / strides[i]) + 1);
    }
    return [inp[0], inp[1], ...spatial];
  }

  if (opName === 'clamp') {
    let shape = [...tensorArgs[0].shape];
    for (let i = 1; i < tensorArgs.length; i++) {
      shape = broadcastShapes(shape, tensorArgs[i].shape) || shape;
    }
    return shape;
  }

  if (opName === 'pad') {
    const inp = tensorArgs[0].shape;
    const low = numberArrayScalar(scalars.low, []);
    const high = numberArrayScalar(scalars.high, []);
    return inp.map((d, i) => d + (low[i] || 0) + (high[i] || 0));
  }

  if (opName === 'one_hot') {
    return [...tensorArgs[0].shape, numberScalar(scalars.depth, 0)];
  }

  if (opName === 'cat') {
    const rank = tensorArgs[0].shape.length;
    const dimArg = numberScalar(scalars.dim, 0);
    const dim = dimArg < 0 ? rank + dimArg : dimArg;
    const shape = [...tensorArgs[0].shape];
    shape[dim] = tensorArgs.reduce((acc, t) => acc + t.shape[dim], 0);
    return shape;
  }

  if (opName === 'stack') {
    const rank = tensorArgs[0].shape.length;
    const dimArg = numberScalar(scalars.dim, 0);
    const dim = dimArg < 0 ? rank + 1 + dimArg : dimArg;
    const shape = [...tensorArgs[0].shape];
    shape.splice(dim, 0, tensorArgs.length);
    return shape;
  }

  if (opName === 'index_select') {
    const shape = [...tensorArgs[0].shape];
    const rank = shape.length;
    const dimArg = numberScalar(scalars.dim, 0);
    const dim = dimArg < 0 ? rank + dimArg : dimArg;
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

export function tensorToContiguous(t: Tensor): NumericTypedArray {
  const srcData = t._impl.storage.data!;
  const srcOff = t._impl.storageOffset;
  const n = t.numel;
  if (t.isContiguous && srcOff === 0 && srcData.length === n) return t.data || srcData;
  const shape = t.shape;
  const strides = t.strides;
  const Ctor = srcData.constructor as { new(length: number): NumericTypedArray };
  const dst = new Ctor(n);
  const writable = dst as MutableNumericArray;
  const ndim = shape.length;
  const indices = new Int32Array(ndim);
  let si = srcOff;
  for (let i = 0; i < n; i++) {
    writable[i] = srcData[si];
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < shape[d]) { si += strides[d]; break; }
      si -= (shape[d] - 1) * strides[d];
      indices[d] = 0;
    }
  }
  return dst;
}

export function wrapResult(data: NumericTypedArray, shape: readonly number[], dtype: DType, device: Device): Tensor {
  const strides = computeStrides(shape);
  const storage = Storage.fromData(data, device);
  const impl = new TensorImpl(storage, 0, shape, strides, dtype, device);
  return new Tensor(impl);
}

function _prod(arr: readonly number[], from: number, to: number): number {
  let p = 1;
  for (let i = from; i < to; i++) p *= arr[i];
  return p;
}

function _hostStack(inputs: readonly NumericTypedArray[], inShape: readonly number[], dim: number, outData: NumericTypedArray) {
  const rank = inShape.length;
  const d = dim < 0 ? rank + 1 + dim : dim;
  const outer = _prod(inShape, 0, d);
  const inner = _prod(inShape, d, rank);
  const n = inputs.length;
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < n; i++) {
      (outData as NumericSettable).set(inputs[i].subarray(o * inner, (o + 1) * inner) as ArrayLike<number | bigint>, (o * n + i) * inner);
    }
  }
}

function _hostCat(inputs: readonly NumericTypedArray[], shapes: readonly (readonly number[])[], dim: number, outData: NumericTypedArray) {
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
      (outData as NumericSettable).set(inputs[k].subarray(o * block, (o + 1) * block) as ArrayLike<number | bigint>, outOff);
      outOff += block;
    }
  }
}

function _runHostConcatLike(opName: string, tensors: readonly Tensor[], scalars: ScalarMap): Tensor {
  const outShape = _inferOutputShape(opName, tensors, scalars);
  const outDtype = tensors[0].dtype;
  const Ctor = typedArrayCtor(outDtype);
  const outData = new Ctor(Math.max(computeNumel(outShape), 1));
  const dim = numberScalar(scalars.dim, 0);
  if (isEagerDeferred() && _gpuConcatFn) {
    const inputArrays = tensors.map(t => _gpuInputArray(t));
    _gpuConcatFn(opName, inputArrays, tensors.map(t => [...t.shape]), dim, outShape, outData, outDtype);
    return wrapResult(outData, outShape, outDtype, tensors[0].device);
  }
  const runtimeArgs = tensors.map(t => tensorToContiguous(t));
  if (opName === 'stack') _hostStack(runtimeArgs, tensors[0].shape, dim, outData);
  else _hostCat(runtimeArgs, tensors.map(t => t.shape), dim, outData);
  return wrapResult(outData, outShape, outDtype, tensors[0].device);
}

let _gpuContiguousFn: GpuContiguousFn | null = null;
export function setGpuContiguousFn(fn: GpuContiguousFn | null) { _gpuContiguousFn = fn; }
export function getGpuContiguousFn() { return _gpuContiguousFn; }
let _gpuConcatFn: GpuConcatFn | null = null;
export function setGpuConcatFn(fn: GpuConcatFn | null) { _gpuConcatFn = fn; }
let _cudnnLSTM: DynamicFn | null = null;
export function setCudnnLSTM(fn: DynamicFn | null) { _cudnnLSTM = fn; }
export function getCudnnLSTM() { return _cudnnLSTM; }
let _cudnnGRU: DynamicFn | null = null;
export function setCudnnGRU(fn: DynamicFn | null) { _cudnnGRU = fn; }
export function getCudnnGRU() { return _cudnnGRU; }
let _webgpuRNN: DynamicFn | null = null;
export function setWebgpuRNN(fn: DynamicFn | null) { _webgpuRNN = fn; }
export function getWebgpuRNN() { return _webgpuRNN; }
let _gpuAdam: DynamicFn | null = null;
export function setGpuAdamFn(fn: DynamicFn | null) { _gpuAdam = fn; }
export function getGpuAdamFn() { return _gpuAdam; }
let _gpuMatmul: DynamicFn | null = null;
export function setGpuMatmul(fn: DynamicFn | null) { _gpuMatmul = fn; }
export function getGpuMatmul() { return _gpuMatmul; }

function _gpuInputArray(t: Tensor): NumericTypedArray {
  if (t.isContiguous && t._impl.storageOffset === 0) {
    const raw = t._impl.storage.rawData;
    if (raw && raw.length === t.numel) return raw;
  }
  if (isEagerDeferred() && _gpuContiguousFn) {
    return _gpuContiguousFn(t._impl.storage.rawData, t.shape, t.strides, t._impl.storageOffset, t.dtype);
  }
  return tensorToContiguous(t);
}
export function gpuContiguousArray(t: Tensor): NumericTypedArray { return _gpuInputArray(t); }

function _wrapOpForJIT(opName: string, dispatchKey: DispatchKeyValue): ((keySet: unknown, ...args: unknown[]) => Tensor) | null {
  const getTarget = _TARGET_FOR_KEY[dispatchKey];
  if (!getTarget) return null;

  const isGPU = dispatchKey === DispatchKey.GPU;
  const isWebGPU = dispatchKey === DispatchKey.CUSTOM_0;
  const hostConcatLike = isGPU && (opName === 'stack' || opName === 'cat');

  return (_keySet: unknown, ...args: unknown[]) => {
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
      _webgpuEagerFn!(entry.compiled, tensors, outData);
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
