import { dispatcher, computeKeySet } from '../../dispatcher/dispatcher.js';
import { getHandle } from './registry.js';
import { scalar } from '../factory/from_ops.js';
import { getGpuMatmul } from '../../dispatcher/jit_dispatch.js';
import type { Tensor } from '../core/tensor.js';

type TensorInput = Tensor | number;

function hasTensorImpl(value: unknown): value is Tensor {
  return typeof value === 'object' && value !== null && '_impl' in value;
}

function _asTensor(value: TensorInput, ref: Tensor): Tensor {
  if (hasTensorImpl(value)) return value;
  return scalar(value, { dtype: ref.dtype, device: ref.device });
}

export function _dispatch(name: string, ...args: unknown[]): unknown {
  const handle = getHandle(name);
  if (!handle) return dispatcher.callOp(name, ...args);
  const ks = computeKeySet(args, handle.schema);
  return dispatcher.dispatch(handle, ks, ...args);
}

function _dispatchTensor(name: string, ...args: unknown[]): Tensor {
  return _dispatch(name, ...args) as Tensor;
}

function _dispatchTensorArray(name: string, ...args: unknown[]): Tensor[] {
  return _dispatch(name, ...args) as Tensor[];
}

export function add(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('add', self, _asTensor(other, self)); }
export function sub(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('sub', self, _asTensor(other, self)); }
export function mul(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('mul', self, _asTensor(other, self)); }
export function div(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('div', self, _asTensor(other, self)); }
export function neg(self: Tensor): Tensor { return _dispatchTensor('neg', self); }
export function pow(self: Tensor, exponent: TensorInput): Tensor { return _dispatchTensor('pow', self, _asTensor(exponent, self)); }
export function remainder(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('rem', self, _asTensor(other, self)); }
export function maximum(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('maximum', self, _asTensor(other, self)); }
export function minimum(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('minimum', self, _asTensor(other, self)); }

export function exp(self: Tensor): Tensor { return _dispatchTensor('exp', self); }
export function log(self: Tensor): Tensor { return _dispatchTensor('log', self); }
export function sqrt(self: Tensor): Tensor { return _dispatchTensor('sqrt', self); }
export function rsqrt(self: Tensor): Tensor { return _dispatchTensor('rsqrt', self); }
export function abs(self: Tensor): Tensor { return _dispatchTensor('abs', self); }
export function sin(self: Tensor): Tensor { return _dispatchTensor('sin', self); }
export function cos(self: Tensor): Tensor { return _dispatchTensor('cos', self); }
export function tanh(self: Tensor): Tensor { return _dispatchTensor('tanh', self); }
export function erf(self: Tensor): Tensor { return _dispatchTensor('erf', self); }
export function erfc(self: Tensor): Tensor { return _dispatchTensor('erfc', self); }
export function lgamma(self: Tensor): Tensor { return _dispatchTensor('lgamma', self); }
export function gamma(self: Tensor): Tensor { return _dispatchTensor('gamma', self); }
export function sigmoid(self: Tensor): Tensor { return _dispatchTensor('sigmoid', self); }
export function relu(self: Tensor): Tensor { return _dispatchTensor('relu', self); }
export function gelu(self: Tensor): Tensor { return _dispatchTensor('gelu', self); }
export function silu(self: Tensor): Tensor { return _dispatchTensor('silu', self); }
export function sign(self: Tensor): Tensor { return _dispatchTensor('sign', self); }
export function floor(self: Tensor): Tensor { return _dispatchTensor('floor', self); }
export function ceil(self: Tensor): Tensor { return _dispatchTensor('ceil', self); }

export function eq(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('eq', self, _asTensor(other, self)); }
export function ne(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('ne', self, _asTensor(other, self)); }
export function lt(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('lt', self, _asTensor(other, self)); }
export function le(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('le', self, _asTensor(other, self)); }
export function gt(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('gt', self, _asTensor(other, self)); }
export function ge(self: Tensor, other: TensorInput): Tensor { return _dispatchTensor('ge', self, _asTensor(other, self)); }
export function where(condition: Tensor, self: Tensor, other: Tensor): Tensor { return _dispatchTensor('where', condition, self, other); }
export function clamp(self: Tensor, min: TensorInput, max: TensorInput): Tensor { return _dispatchTensor('clamp', self, _asTensor(min, self), _asTensor(max, self)); }
export function pad(self: Tensor, low: readonly number[], high: readonly number[], value: TensorInput = 0): Tensor { return _dispatchTensor('pad', self, _asTensor(value, self), low, high); }
export function one_hot(indices: Tensor, depth: number): Tensor { return _dispatchTensor('one_hot', indices, depth); }
export function index_select(self: Tensor, dim: number, index: Tensor): Tensor { return _dispatchTensor('index_select', self, index, dim); }
export function gather(self: Tensor, dim: number, index: Tensor): Tensor { return _dispatchTensor('gather', self, index, dim); }
export function scatter_add(self: Tensor, dim: number, index: Tensor, src: Tensor): Tensor { return _dispatchTensor('scatter_add', self, index, src, dim); }
export function scatter(self: Tensor, dim: number, index: Tensor, src: Tensor): Tensor { return _dispatchTensor('scatter', self, dim, index, src); }

export function sum(self: Tensor, dim?: number | readonly number[] | null, keepdim?: boolean): Tensor { return _dispatchTensor('sum', self, dim, keepdim); }
export function mean(self: Tensor, dim?: number | readonly number[] | null, keepdim?: boolean): Tensor { return _dispatchTensor('mean', self, dim, keepdim); }
export function max(self: Tensor, dim?: number | readonly number[] | null, keepdim?: boolean): Tensor { return _dispatchTensor('max', self, dim, keepdim); }
export function min(self: Tensor, dim?: number | readonly number[] | null, keepdim?: boolean): Tensor { return _dispatchTensor('min', self, dim, keepdim); }
export function argmax(self: Tensor, dim?: number | null, keepdim?: boolean): Tensor { return _dispatchTensor('argmax', self, dim, keepdim); }
export function argmin(self: Tensor, dim?: number | null, keepdim?: boolean): Tensor { return _dispatchTensor('argmin', self, dim, keepdim); }
export function prod(self: Tensor, dim?: number | readonly number[] | null, keepdim?: boolean): Tensor { return _dispatchTensor('prod', self, dim, keepdim); }

export function matmul(self: Tensor, other: Tensor): Tensor {
  const h = getGpuMatmul();
  if (h) { const r = h(self, other); if (r !== null) return r as Tensor; }
  return _dispatchTensor('matmul', self, other);
}
export function dot(self: Tensor, other: Tensor): Tensor { return _dispatchTensor('dot', self, other); }

export function cat(tensors: readonly Tensor[], dim: number): Tensor { return _dispatchTensor('cat', tensors, dim); }
export function stack(tensors: readonly Tensor[], dim: number): Tensor { return _dispatchTensor('stack', tensors, dim); }

export function clone(self: Tensor): Tensor { return _dispatchTensor('clone', self); }
export function fill(self: Tensor, value: number | bigint): Tensor { return _dispatchTensor('fill', self, value); }

export function reshape(self: Tensor, shape: readonly number[]): Tensor { return _dispatchTensor('reshape', self, shape); }
export function transpose(self: Tensor, dim0: number, dim1: number): Tensor { return _dispatchTensor('transpose', self, dim0, dim1); }
export function permute(self: Tensor, dims: readonly number[]): Tensor { return _dispatchTensor('permute', self, dims); }
export function broadcast_in_dim(self: Tensor, resultShape: readonly number[], broadcastDimensions: readonly number[]): Tensor { return _dispatchTensor('broadcast_in_dim', self, resultShape, broadcastDimensions); }
export function expand(self: Tensor, shape: readonly number[]): Tensor { return _dispatchTensor('expand', self, shape); }
export function slice(self: Tensor, dim: number, start: number, end: number | null = null, step = 1): Tensor { return _dispatchTensor('slice', self, dim, start, end, step); }
export function unsqueeze(self: Tensor, dim: number): Tensor { return _dispatchTensor('unsqueeze', self, dim); }
export function squeeze(self: Tensor, dim: number | null = null): Tensor { return _dispatchTensor('squeeze', self, dim); }
export function narrow(self: Tensor, dim: number, start: number, length: number): Tensor { return _dispatchTensor('narrow', self, dim, start, length); }
export function select(self: Tensor, dim: number, index: number): Tensor { return _dispatchTensor('select', self, dim, index); }
export function contiguous(self: Tensor): Tensor { return _dispatchTensor('contiguous', self); }
export function repeat(self: Tensor, reps: readonly number[]): Tensor { return _dispatchTensor('repeat', self, reps); }
export function tile(self: Tensor, reps: readonly number[]): Tensor { return _dispatchTensor('tile', self, reps); }
export function split(self: Tensor, sizeOrSizes: number | readonly number[], dim = 0): Tensor[] {
  if (typeof sizeOrSizes !== 'number') return _dispatchTensorArray('split', self, sizeOrSizes, dim);
  const rank = self.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  const n = self.shape[d];
  const sizes: number[] = [];
  for (let off = 0; off < n; off += sizeOrSizes) sizes.push(Math.min(sizeOrSizes, n - off));
  return _dispatchTensorArray('split', self, sizes, dim);
}
export function chunk(self: Tensor, chunks: number, dim = 0): Tensor[] { return _dispatchTensorArray('chunk', self, chunks, dim); }
export function roll(self: Tensor, shift: number, dim = 0): Tensor { return _dispatchTensor('roll', self, shift, dim); }
export function flip(self: Tensor, dims: number | readonly number[]): Tensor { return _dispatchTensor('flip', self, Array.isArray(dims) ? dims : [dims]); }
export function cumsum(self: Tensor, dim = 0): Tensor { return _dispatchTensor('cumsum', self, dim); }
export function sort(self: Tensor, dim = -1, descending = false): Tensor { return _dispatchTensor('sort', self, dim, descending); }
export function argsort(self: Tensor, dim = -1, descending = false): Tensor { return _dispatchTensor('argsort', self, dim, descending); }
export function topk(self: Tensor, k: number, dim = -1, largest = true): Tensor[] { return _dispatchTensorArray('topk', self, k, dim, largest); }

export function softmax(self: Tensor, dim: number): Tensor { return _dispatchTensor('softmax', self, dim); }
export function log_softmax(self: Tensor, dim: number): Tensor { return _dispatchTensor('log_softmax', self, dim); }
export function layer_norm(input: Tensor, weight: Tensor, bias: Tensor, axis: number, eps: number): Tensor { return _dispatchTensor('layer_norm', input, weight, bias, axis, eps); }
export function batch_norm(input: Tensor, weight: Tensor, bias: Tensor, mean: Tensor, variance: Tensor, axis: number, eps: number): Tensor { return _dispatchTensor('batch_norm', input, weight, bias, mean, variance, axis, eps); }
export function conv2d(input: Tensor, weight: Tensor, strides: readonly number[], padding: readonly number[], dilation: readonly number[], groups: number): Tensor { return _dispatchTensor('conv2d', input, weight, strides, padding, dilation, groups); }
export function pool2d(input: Tensor, poolType: string, kernelSize: readonly number[], strides: readonly number[], padding: readonly number[]): Tensor { return _dispatchTensor('pool2d', input, poolType, kernelSize, strides, padding); }
export function embedding(weight: Tensor, indices: Tensor): Tensor { return _dispatchTensor('embedding', weight, indices); }
