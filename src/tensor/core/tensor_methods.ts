import type * as ops from '../ops/ops.js';
import type { Device } from '../types/device.js';
import type { Tensor } from './tensor.js';

export const TENSOR_OP_METHODS = [
  'neg', 'exp', 'log', 'sqrt', 'rsqrt', 'abs', 'sin', 'cos', 'tanh',
  'erf', 'erfc', 'lgamma', 'gamma', 'sigmoid', 'relu', 'gelu', 'silu',
  'sign', 'floor', 'ceil', 'clone', 'contiguous',

  'add', 'sub', 'mul', 'div', 'pow', 'remainder', 'maximum', 'minimum',
  'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'matmul', 'dot', 'flip', 'unsqueeze',

  'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'prod',

  'softmax', 'log_softmax', 'roll', 'cumsum', 'sort', 'argsort',
  'topk', 'split', 'chunk', 'squeeze',

  'gather', 'scatter_add', 'scatter', 'transpose', 'slice', 'narrow', 'select',
] as const;

export const TENSOR_SHAPE_METHODS = ['reshape', 'permute', 'expand', 'repeat', 'tile'] as const;

export const TENSOR_SUGAR_METHODS = ['to', 'mm', 't', 'requires_grad'] as const;

export type TensorOpMethodName = (typeof TENSOR_OP_METHODS)[number];
export type TensorShapeMethodName = (typeof TENSOR_SHAPE_METHODS)[number];
export type TensorSugarMethodName = (typeof TENSOR_SUGAR_METHODS)[number];

type WithoutSelf<F> = F extends (self: Tensor, ...rest: infer A) => infer R ? (...args: A) => R : never;

type DerivedMethods = { [K in TensorOpMethodName]: WithoutSelf<(typeof ops)[K]> };

type ShapeMethod = {
  (shape: readonly number[]): Tensor;
  (...shape: number[]): Tensor;
};

type ShapeMethods = { [K in TensorShapeMethodName]: ShapeMethod };

export interface TensorSugarMethods {
  to(device: Device): Tensor;
  mm(other: Tensor): Tensor;
  t(): Tensor;
  requires_grad(flag?: boolean): Tensor;
}

export interface TensorMethods extends DerivedMethods, ShapeMethods, TensorSugarMethods {}

type SameKeys<A, B> = [A] extends [B] ? ([B] extends [A] ? true : never) : never;

export const SUGAR_NAMES_MATCH_DECLARATIONS: SameKeys<TensorSugarMethodName, keyof TensorSugarMethods> = true;
