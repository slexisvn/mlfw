import * as ops from './ops.js';
import { fromBuffer } from '../factory/from_ops.js';
import type { Tensor } from '../core/tensor.js';
import type { Device } from '../types/device.js';

type TensorMethod = (this: Tensor, ...args: unknown[]) => unknown;
type TensorProto = Record<string, unknown>;
type TensorClass = {
  prototype: TensorProto;
};
type OpsModule = Record<string, (...args: unknown[]) => unknown>;

const OPS = ops as OpsModule;

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

const METHOD_DEFAULTS: Record<string, readonly unknown[]> = {
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

function withDefaults(args: readonly unknown[], defaults: readonly unknown[]): unknown[] {
  const out: unknown[] = [];
  for (let i = 0; i < defaults.length; i++) out[i] = args[i] === undefined ? defaults[i] : args[i];
  for (let i = defaults.length; i < args.length; i++) out[i] = args[i];
  return out;
}

function installSelfMethods(proto: TensorProto): void {
  for (const name of SELF_METHODS) {
    proto[name] = function() { return OPS[name](this); };
  }
}

function installOneArgMethods(proto: TensorProto): void {
  for (const name of ONE_ARG_METHODS) {
    proto[name] = function(arg: unknown) { return OPS[name](this, arg); };
  }
}

function installDefaultedMethods(proto: TensorProto): void {
  for (const [name, defaults] of Object.entries(METHOD_DEFAULTS)) {
    proto[name] = function(...args: unknown[]) { return OPS[name](this, ...withDefaults(args, defaults)); };
  }
}

function installReductionMethods(proto: TensorProto): void {
  for (const name of REDUCTION_METHODS) {
    proto[name] = function(dim: unknown, keepdim: unknown) { return OPS[name](this, dim, keepdim); };
  }
}

function arrayArg(args: readonly unknown[]): readonly unknown[] {
  return args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
}

export function installOps(TensorClass: TensorClass): void {
  const proto = TensorClass.prototype;

  proto.to = function(this: Tensor, device: Device) {
    if (this.device.equals(device)) return this;
    const source = OPS.contiguous(this) as Tensor;
    const data = source.data!.slice(0, this.numel);
    return fromBuffer(data, this.shape, this.dtype, { device });
  };

  installSelfMethods(proto);
  installOneArgMethods(proto);
  installDefaultedMethods(proto);
  installReductionMethods(proto);

  proto.mm = function(this: Tensor, other: unknown) { return ops.matmul(this, other as Tensor); };
  proto.requires_grad = function(this: Tensor, flag = true) { return this.requiresGrad_(flag as boolean); };
  proto.gather = function(this: Tensor, dim: unknown, index: unknown) { return ops.gather(this, dim as number, index as Tensor); };
  proto.scatter_add = function(this: Tensor, dim: unknown, index: unknown, src: unknown) { return ops.scatter_add(this, dim as number, index as Tensor, src as Tensor); };
  proto.scatter = function(this: Tensor, dim: unknown, index: unknown, src: unknown) { return ops.scatter(this, dim as number, index as Tensor, src as Tensor); };
  proto.transpose = function(this: Tensor, d0: unknown, d1: unknown) { return ops.transpose(this, d0 as number, d1 as number); };
  proto.slice = function(this: Tensor, dim: unknown, start: unknown, end: unknown, step: unknown) { return ops.slice(this, dim as number, start as number, end as number | null, step as number); };
  proto.narrow = function(this: Tensor, dim: unknown, start: unknown, length: unknown) { return ops.narrow(this, dim as number, start as number, length as number); };
  proto.select = function(this: Tensor, dim: unknown, index: unknown) { return ops.select(this, dim as number, index as number); };

  proto.reshape = function(this: Tensor, ...args: unknown[]) { return ops.reshape(this, arrayArg(args) as readonly number[]); };
  proto.permute = function(this: Tensor, ...args: unknown[]) { return ops.permute(this, arrayArg(args) as readonly number[]); };
  proto.expand = function(this: Tensor, ...args: unknown[]) { return ops.expand(this, arrayArg(args) as readonly number[]); };
  proto.repeat = function(this: Tensor, ...args: unknown[]) { return ops.repeat(this, arrayArg(args) as readonly number[]); };
  proto.tile = function(this: Tensor, ...args: unknown[]) { return ops.tile(this, arrayArg(args) as readonly number[]); };

  proto.t = function(this: Tensor) {
    if (this.ndim !== 2) throw new Error('t() expects a 2D tensor');
    return ops.transpose(this, 0, 1);
  };
}
