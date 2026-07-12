import * as ops from './ops.js';
import { fromBuffer } from '../factory/from_ops.js';

const SELF_METHODS = [
  'neg', 'exp', 'log', 'sqrt', 'rsqrt', 'abs', 'sin', 'cos', 'tanh',
  'erf', 'erfc', 'lgamma', 'gamma', 'sigmoid', 'relu', 'gelu', 'silu',
  'sign', 'floor', 'ceil', 'clone', 'contiguous',
];

const ONE_ARG_METHODS = [
  'add', 'sub', 'mul', 'div', 'pow', 'remainder', 'maximum', 'minimum',
  'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'matmul', 'dot', 'flip', 'unsqueeze',
];

const REDUCTION_METHODS = ['sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'prod'];

const METHOD_DEFAULTS = {
  softmax: [-1],
  log_softmax: [-1],
  roll: [undefined, 0],
  cumsum: [0],
  sort: [-1, false],
  argsort: [-1, false],
  topk: [undefined, -1, true],
  split: [undefined, 0],
  chunk: [undefined, 0],
  squeeze: [null],
};

function withDefaults(args, defaults) {
  const out = [];
  for (let i = 0; i < defaults.length; i++) out[i] = args[i] === undefined ? defaults[i] : args[i];
  for (let i = defaults.length; i < args.length; i++) out[i] = args[i];
  return out;
}

function installSelfMethods(proto) {
  for (const name of SELF_METHODS) {
    proto[name] = function() { return ops[name](this); };
  }
}

function installOneArgMethods(proto) {
  for (const name of ONE_ARG_METHODS) {
    proto[name] = function(arg) { return ops[name](this, arg); };
  }
}

function installDefaultedMethods(proto) {
  for (const [name, defaults] of Object.entries(METHOD_DEFAULTS)) {
    proto[name] = function(...args) { return ops[name](this, ...withDefaults(args, defaults)); };
  }
}

function installReductionMethods(proto) {
  for (const name of REDUCTION_METHODS) {
    proto[name] = function(dim, keepdim) { return ops[name](this, dim, keepdim); };
  }
}

function arrayArg(args) {
  return args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
}

export function installOps(TensorClass) {
  const proto = TensorClass.prototype;

  proto.to = function(device) {
    if (this.device.equals(device)) return this;
    const source = this.contiguous();
    const data = source.data.slice(0, this.numel);
    return fromBuffer(data, this.shape, this.dtype, { device });
  };

  installSelfMethods(proto);
  installOneArgMethods(proto);
  installDefaultedMethods(proto);
  installReductionMethods(proto);

  proto.mm = function(other) { return ops.matmul(this, other); };
  proto.requires_grad = function(flag = true) { return this.requiresGrad_(flag); };
  proto.gather = function(dim, index) { return ops.gather(this, dim, index); };
  proto.scatter_add = function(dim, index, src) { return ops.scatter_add(this, dim, index, src); };
  proto.scatter = function(dim, index, src) { return ops.scatter(this, dim, index, src); };
  proto.transpose = function(d0, d1) { return ops.transpose(this, d0, d1); };
  proto.slice = function(dim, start, end, step) { return ops.slice(this, dim, start, end, step); };
  proto.narrow = function(dim, start, length) { return ops.narrow(this, dim, start, length); };
  proto.select = function(dim, index) { return ops.select(this, dim, index); };

  proto.reshape = function(...args) { return ops.reshape(this, arrayArg(args)); };
  proto.permute = function(...args) { return ops.permute(this, arrayArg(args)); };
  proto.expand = function(...args) { return ops.expand(this, arrayArg(args)); };
  proto.repeat = function(...args) { return ops.repeat(this, arrayArg(args)); };
  proto.tile = function(...args) { return ops.tile(this, arrayArg(args)); };

  proto.t = function() {
    if (this.ndim !== 2) throw new Error('t() expects a 2D tensor');
    return ops.transpose(this, 0, 1);
  };
}
