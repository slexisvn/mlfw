import * as ops from './ops.js';
import { fromBuffer } from '../factory/from_ops.js';
import { TENSOR_OP_METHODS, TENSOR_SHAPE_METHODS } from '../core/tensor_methods.js';
import type { Tensor } from '../core/tensor.js';
import type { Device } from '../types/device.js';

type TensorProto = Record<string, unknown>;
type TensorClass = {
  prototype: TensorProto;
};
type OpsModule = Record<string, (...args: unknown[]) => unknown>;

const OPS = ops as unknown as OpsModule;

const RANK_FOR_T = 2;

function shapeArg(args: readonly unknown[]): readonly number[] {
  const sugar = args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
  return sugar as readonly number[];
}

export function installOps(TensorClass: TensorClass): void {
  const proto = TensorClass.prototype;

  for (const name of TENSOR_OP_METHODS) {
    proto[name] = function(this: Tensor, ...args: unknown[]) { return OPS[name](this, ...args); };
  }

  for (const name of TENSOR_SHAPE_METHODS) {
    proto[name] = function(this: Tensor, ...args: unknown[]) { return OPS[name](this, shapeArg(args)); };
  }

  proto.to = function(this: Tensor, device: Device) {
    if (this.device.equals(device)) return this;
    const source = ops.contiguous(this);
    const data = source.data!.slice(0, this.numel);
    return fromBuffer(data, this.shape, this.dtype, { device });
  };

  proto.mm = function(this: Tensor, other: Tensor) { return ops.matmul(this, other); };

  proto.requires_grad = function(this: Tensor, flag = true) { return this.requiresGrad_(flag); };

  proto.t = function(this: Tensor) {
    if (this.ndim !== RANK_FOR_T) throw new Error('t() expects a 2D tensor');
    return ops.transpose(this, 0, 1);
  };
}
