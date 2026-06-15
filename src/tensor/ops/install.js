import * as ops from './ops.js';
import * as composite from './composite.js';
import { installViewOps } from '../view/view_ops.js';
import { fromBuffer } from '../factory/from_ops.js';

export function installOps(TensorClass) {
  const proto = TensorClass.prototype;

  proto.to = function(device) {
    if (this.device.equals(device)) return this;
    const source = this.contiguous();
    const data = source.data.slice(0, this.numel);
    return fromBuffer(data, this.shape, this.dtype, { device });
  };

  proto.add = function(other) { return ops.add(this, other); };
  proto.sub = function(other) { return ops.sub(this, other); };
  proto.mul = function(other) { return ops.mul(this, other); };
  proto.div = function(other) { return ops.div(this, other); };
  proto.neg = function() { return ops.neg(this); };
  proto.pow = function(exp) { return ops.pow(this, exp); };
  proto.remainder = function(other) { return ops.remainder(this, other); };
  proto.maximum = function(other) { return ops.maximum(this, other); };
  proto.minimum = function(other) { return ops.minimum(this, other); };

  proto.exp = function() { return ops.exp(this); };
  proto.log = function() { return ops.log(this); };
  proto.sqrt = function() { return ops.sqrt(this); };
  proto.rsqrt = function() { return ops.rsqrt(this); };
  proto.abs = function() { return ops.abs(this); };
  proto.sin = function() { return ops.sin(this); };
  proto.cos = function() { return ops.cos(this); };
  proto.tanh = function() { return ops.tanh(this); };
  proto.sigmoid = function() { return ops.sigmoid(this); };
  proto.relu = function() { return ops.relu(this); };
  proto.gelu = function() { return ops.gelu(this); };
  proto.silu = function() { return ops.silu(this); };
  proto.sign = function() { return ops.sign(this); };
  proto.floor = function() { return ops.floor(this); };
  proto.ceil = function() { return ops.ceil(this); };

  proto.eq = function(other) { return ops.eq(this, other); };
  proto.ne = function(other) { return ops.ne(this, other); };
  proto.lt = function(other) { return ops.lt(this, other); };
  proto.le = function(other) { return ops.le(this, other); };
  proto.gt = function(other) { return ops.gt(this, other); };
  proto.ge = function(other) { return ops.ge(this, other); };

  proto.sum = function(dim, keepdim) { return ops.sum(this, dim, keepdim); };
  proto.mean = function(dim, keepdim) { return ops.mean(this, dim, keepdim); };
  proto.max = function(dim, keepdim) { return ops.max(this, dim, keepdim); };
  proto.min = function(dim, keepdim) { return ops.min(this, dim, keepdim); };
  proto.argmax = function(dim, keepdim) { return ops.argmax(this, dim, keepdim); };
  proto.argmin = function(dim, keepdim) { return ops.argmin(this, dim, keepdim); };
  proto.prod = function(dim, keepdim) { return ops.prod(this, dim, keepdim); };

  proto.matmul = function(other) { return ops.matmul(this, other); };
  proto.dot = function(other) { return ops.dot(this, other); };
  proto.mm = function(other) { return ops.matmul(this, other); };

  proto.softmax = function(dim = -1) { return ops.softmax(this, dim); };
  proto.log_softmax = function(dim = -1) { return ops.log_softmax(this, dim); };

  proto.requires_grad = function(flag = true) { return this.requiresGrad_(flag); };

  proto.clone = function() { return ops.clone(this); };

  proto.roll = function(shift, dim = 0) { return composite.roll(this, shift, dim); };
  proto.flip = function(dims) { return composite.flip(this, dims); };
  proto.cumsum = function(dim = 0) { return composite.cumsum(this, dim); };
  proto.sort = function(dim = -1, descending = false) { return composite.sort(this, dim, descending); };
  proto.topk = function(k, dim = -1, largest = true) { return composite.topk(this, k, dim, largest); };
  proto.argsort = function(dim = -1, descending = false) { return composite.argsort(this, dim, descending); };
  proto.gather = function(dim, index) { return ops.gather(this, dim, index); };
  proto.scatter_add = function(dim, index, src) { return ops.scatter_add(this, dim, index, src); };
  proto.scatter = function(dim, index, src) { return composite.scatter(this, dim, index, src); };

  installViewOps(TensorClass);
}
