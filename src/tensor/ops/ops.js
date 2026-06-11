import { dispatcher, computeKeySet } from '../../dispatcher/dispatcher.js';
import { getHandle } from './registry.js';
import { scalar } from '../factory/from_ops.js';

function _asTensor(value, ref) {
  if (value && value._impl) return value;
  return scalar(value, { dtype: ref.dtype });
}

function _dispatch(name, ...args) {
  const handle = getHandle(name);
  if (!handle) return dispatcher.callOp(name, ...args);
  const ks = computeKeySet(args, handle.schema);
  return dispatcher.dispatch(handle, ks, ...args);
}

export function add(self, other) { return _dispatch('add', self, other); }
export function sub(self, other) { return _dispatch('sub', self, other); }
export function mul(self, other) { return _dispatch('mul', self, other); }
export function div(self, other) { return _dispatch('div', self, other); }
export function neg(self) { return _dispatch('neg', self); }
export function pow(self, exponent) { return _dispatch('pow', self, exponent); }
export function remainder(self, other) { return _dispatch('rem', self, other); }
export function maximum(self, other) { return _dispatch('maximum', self, other); }
export function minimum(self, other) { return _dispatch('minimum', self, other); }

export function exp(self) { return _dispatch('exp', self); }
export function log(self) { return _dispatch('log', self); }
export function sqrt(self) { return _dispatch('sqrt', self); }
export function rsqrt(self) { return _dispatch('rsqrt', self); }
export function abs(self) { return _dispatch('abs', self); }
export function sin(self) { return _dispatch('sin', self); }
export function cos(self) { return _dispatch('cos', self); }
export function tanh(self) { return _dispatch('tanh', self); }
export function sigmoid(self) { return _dispatch('sigmoid', self); }
export function relu(self) { return _dispatch('relu', self); }
export function gelu(self) { return _dispatch('gelu', self); }
export function silu(self) { return _dispatch('silu', self); }
export function sign(self) { return _dispatch('sign', self); }
export function floor(self) { return _dispatch('floor', self); }
export function ceil(self) { return _dispatch('ceil', self); }

export function eq(self, other) { return _dispatch('eq', self, other); }
export function ne(self, other) { return _dispatch('ne', self, other); }
export function lt(self, other) { return _dispatch('lt', self, other); }
export function le(self, other) { return _dispatch('le', self, other); }
export function gt(self, other) { return _dispatch('gt', self, other); }
export function ge(self, other) { return _dispatch('ge', self, other); }
export function where(condition, self, other) { return _dispatch('where', condition, self, other); }
export function clamp(self, min, max) { return _dispatch('clamp', self, _asTensor(min, self), _asTensor(max, self)); }
export function pad(self, low, high, value = 0) { return _dispatch('pad', self, _asTensor(value, self), low, high); }
export function one_hot(indices, depth) { return _dispatch('one_hot', indices, depth); }
export function index_select(self, dim, index) { return _dispatch('index_select', self, index, dim); }
export function gather(self, dim, index) { return _dispatch('gather', self, index, dim); }
export function scatter_add(self, dim, index, src) { return _dispatch('scatter_add', self, index, src, dim); }

export function sum(self, dim, keepdim) { return _dispatch('sum', self, dim, keepdim); }
export function mean(self, dim, keepdim) { return _dispatch('mean', self, dim, keepdim); }
export function max(self, dim, keepdim) { return _dispatch('max', self, dim, keepdim); }
export function min(self, dim, keepdim) { return _dispatch('min', self, dim, keepdim); }
export function argmax(self, dim, keepdim) { return _dispatch('argmax', self, dim, keepdim); }
export function argmin(self, dim, keepdim) { return _dispatch('argmin', self, dim, keepdim); }
export function prod(self, dim, keepdim) { return _dispatch('prod', self, dim, keepdim); }

export function matmul(self, other) { return _dispatch('matmul', self, other); }
export function dot(self, other) { return _dispatch('dot', self, other); }

export function cat(tensors, dim) { return _dispatch('cat', tensors, dim); }
export function stack(tensors, dim) { return _dispatch('stack', tensors, dim); }

export function clone(self) { return _dispatch('clone', self); }
export function fill(self, value) { return _dispatch('fill', self, value); }

export function transpose_op(self, dim0, dim1) { return _dispatch('transpose', self, dim0, dim1); }

export function softmax(self, dim) { return _dispatch('softmax', self, dim); }
export function log_softmax(self, dim) { return _dispatch('log_softmax', self, dim); }
export function layer_norm(input, weight, bias, axis, eps) { return _dispatch('layer_norm', input, weight, bias, axis, eps); }
export function batch_norm(input, weight, bias, mean, variance, axis, eps) { return _dispatch('batch_norm', input, weight, bias, mean, variance, axis, eps); }
export function conv2d(input, weight, strides, padding, dilation, groups) { return _dispatch('conv2d', input, weight, strides, padding, dilation, groups); }
export function pool2d(input, poolType, kernelSize, strides, padding) { return _dispatch('pool2d', input, poolType, kernelSize, strides, padding); }
export function embedding(weight, indices) { return _dispatch('embedding', weight, indices); }
