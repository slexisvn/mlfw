import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';
import type { NNTensor, OptionalTensor } from '../types.js';
import type { NumericTypedArray } from '../../tensor/types/dtype.js';

type NumericMutableArray = Exclude<NumericTypedArray, BigInt64Array>;

export function layer_norm(input: NNTensor, normalizedShape: readonly number[], weight: OptionalTensor, bias: OptionalTensor, eps = 1e-5): NNTensor {
  if ((input instanceof SymbolicTensor || input.isSymbolic) && weight && bias) {
    const axis = input.ndim - normalizedShape.length;
    return ops.layer_norm(input, weight, bias, axis, eps) as NNTensor;
  }
  const axis = input.ndim - normalizedShape.length;
  const dims = [];
  for (let i = axis; i < input.ndim; i++) dims.push(i);

  let meanVal: NNTensor = input;
  for (let i = dims.length - 1; i >= 0; i--) {
    meanVal = ops.mean(meanVal, dims[i], true) as NNTensor;
  }

  const centered = ops.sub(input, meanVal) as NNTensor;
  const sq = ops.mul(centered, centered) as NNTensor;

  let variance: NNTensor = sq;
  for (let i = dims.length - 1; i >= 0; i--) {
    variance = ops.mean(variance, dims[i], true) as NNTensor;
  }

  const epsT = full([], eps);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(variance, epsT))) as NNTensor;
  let normalized = ops.mul(centered, invStd) as NNTensor;

  if (weight) normalized = ops.mul(normalized, weight) as NNTensor;
  if (bias) normalized = ops.add(normalized, bias) as NNTensor;
  return normalized;
}

export function group_norm(input: NNTensor, numGroups: number, weight: OptionalTensor, bias: OptionalTensor, eps = 1e-5): NNTensor {
  const shape = input.shape;
  const N = shape[0];
  const C = shape[1];
  const spatial = shape.slice(2);
  const grouped = input.reshape([N, numGroups, C / numGroups, ...spatial]);
  const dims = [];
  for (let i = 2; i < grouped.ndim; i++) dims.push(i);

  let meanVal: NNTensor = grouped;
  for (let i = dims.length - 1; i >= 0; i--) meanVal = ops.mean(meanVal, dims[i], true) as NNTensor;
  const centered = ops.sub(grouped, meanVal) as NNTensor;
  const sq = ops.mul(centered, centered) as NNTensor;
  let variance: NNTensor = sq;
  for (let i = dims.length - 1; i >= 0; i--) variance = ops.mean(variance, dims[i], true) as NNTensor;

  const epsT = full([], eps);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(variance, epsT))) as NNTensor;
  let normalized = (ops.mul(centered, invStd) as NNTensor).reshape(shape);

  const affineShape = [1, C, ...spatial.map(() => 1)];
  if (weight) normalized = ops.mul(normalized, weight.reshape(affineShape)) as NNTensor;
  if (bias) normalized = ops.add(normalized, bias.reshape(affineShape)) as NNTensor;
  return normalized;
}

export function rms_norm(input: NNTensor, normalizedShape: readonly number[], weight: OptionalTensor, eps = 1e-6): NNTensor {
  const axis = input.ndim - normalizedShape.length;
  const dims = [];
  for (let i = axis; i < input.ndim; i++) dims.push(i);

  let meanSq: NNTensor = ops.mul(input, input) as NNTensor;
  for (let i = dims.length - 1; i >= 0; i--) meanSq = ops.mean(meanSq, dims[i], true) as NNTensor;

  const invRms = ops.div(full([], 1), ops.sqrt(ops.add(meanSq, full([], eps)))) as NNTensor;
  const normalized = ops.mul(input, invRms) as NNTensor;
  return weight ? ops.mul(normalized, weight) as NNTensor : normalized;
}

export function instance_norm(input: NNTensor, weight: OptionalTensor, bias: OptionalTensor, eps = 1e-5): NNTensor {
  return group_norm(input, input.shape[1], weight, bias, eps);
}

const CHANNEL_AXIS = 1;

function channelShape(ndim: number, channels: number): number[] {
  const shape = new Array(ndim).fill(1);
  shape[CHANNEL_AXIS] = channels;
  return shape;
}

function reduceMeanOver(value: NNTensor, dims: readonly number[]): NNTensor {
  let result: NNTensor = value;
  for (let i = dims.length - 1; i >= 0; i--) {
    result = ops.mean(result, dims[i], true) as NNTensor;
  }
  return result;
}

function blendRunning(buffer: NNTensor, stat: NNTensor, momentum: number): void {
  const data = buffer.data as NumericMutableArray | null;
  if (!data) return;
  const next = stat.reshape([buffer.shape[0]]).toArray() as ArrayLike<number>;
  for (let i = 0; i < data.length; i++) {
    data[i] = data[i] * (1 - momentum) + next[i] * momentum;
  }
}

export function batch_norm(input: NNTensor, runningMean: OptionalTensor, runningVar: OptionalTensor, weight: OptionalTensor, bias: OptionalTensor, training = true, eps = 1e-5, momentum = 0.1): NNTensor {
  const symbolic = input instanceof SymbolicTensor || input.isSymbolic;
  if (!training) {
    return ops.batch_norm(input, weight as unknown as NNTensor, bias as unknown as NNTensor, runningMean as unknown as NNTensor, runningVar as unknown as NNTensor, CHANNEL_AXIS, eps) as NNTensor;
  }

  const dims = [];
  for (let i = 0; i < input.ndim; i++) {
    if (i !== CHANNEL_AXIS) dims.push(i);
  }

  const batchMean = reduceMeanOver(input, dims);
  const centered = ops.sub(input, batchMean) as NNTensor;
  const batchVar = reduceMeanOver(ops.mul(centered, centered) as NNTensor, dims);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(batchVar, full([], eps)))) as NNTensor;
  let normalized = ops.mul(centered, invStd) as NNTensor;

  const affineShape = channelShape(input.ndim, input.shape[CHANNEL_AXIS]);
  if (weight) normalized = ops.mul(normalized, weight.reshape(affineShape)) as NNTensor;
  if (bias) normalized = ops.add(normalized, bias.reshape(affineShape)) as NNTensor;

  if (!symbolic) {
    if (runningMean) blendRunning(runningMean, batchMean, momentum);
    if (runningVar) blendRunning(runningVar, batchVar, momentum);
  }
  return normalized;
}
