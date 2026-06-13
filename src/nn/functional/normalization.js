import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';

export function layer_norm(input, normalizedShape, weight, bias, eps = 1e-5) {
  if ((input instanceof SymbolicTensor || input.isSymbolic) && weight && bias) {
    const axis = input.ndim - normalizedShape.length;
    return ops.layer_norm(input, weight, bias, axis, eps);
  }
  const axis = input.ndim - normalizedShape.length;
  const dims = [];
  for (let i = axis; i < input.ndim; i++) dims.push(i);

  let meanVal = input;
  for (let i = dims.length - 1; i >= 0; i--) {
    meanVal = ops.mean(meanVal, dims[i], true);
  }

  const centered = ops.sub(input, meanVal);
  const sq = ops.mul(centered, centered);

  let variance = sq;
  for (let i = dims.length - 1; i >= 0; i--) {
    variance = ops.mean(variance, dims[i], true);
  }

  const epsT = full([], eps);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(variance, epsT)));
  let normalized = ops.mul(centered, invStd);

  if (weight) normalized = ops.mul(normalized, weight);
  if (bias) normalized = ops.add(normalized, bias);
  return normalized;
}

export function group_norm(input, numGroups, weight, bias, eps = 1e-5) {
  const shape = input.shape;
  const N = shape[0];
  const C = shape[1];
  const spatial = shape.slice(2);
  const grouped = input.reshape([N, numGroups, C / numGroups, ...spatial]);
  const dims = [];
  for (let i = 2; i < grouped.ndim; i++) dims.push(i);

  let meanVal = grouped;
  for (let i = dims.length - 1; i >= 0; i--) meanVal = ops.mean(meanVal, dims[i], true);
  const centered = ops.sub(grouped, meanVal);
  const sq = ops.mul(centered, centered);
  let variance = sq;
  for (let i = dims.length - 1; i >= 0; i--) variance = ops.mean(variance, dims[i], true);

  const epsT = full([], eps);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(variance, epsT)));
  let normalized = ops.mul(centered, invStd).reshape(shape);

  const affineShape = [1, C, ...spatial.map(() => 1)];
  if (weight) normalized = ops.mul(normalized, weight.reshape(affineShape));
  if (bias) normalized = ops.add(normalized, bias.reshape(affineShape));
  return normalized;
}

const CHANNEL_AXIS = 1;

function channelShape(ndim, channels) {
  const shape = new Array(ndim).fill(1);
  shape[CHANNEL_AXIS] = channels;
  return shape;
}

function reduceMeanOver(value, dims) {
  let result = value;
  for (let i = dims.length - 1; i >= 0; i--) {
    result = ops.mean(result, dims[i], true);
  }
  return result;
}

function blendRunning(buffer, stat, momentum) {
  const data = buffer.data;
  if (!data) return;
  const next = stat.reshape([buffer.shape[0]]).toArray();
  for (let i = 0; i < data.length; i++) {
    data[i] = data[i] * (1 - momentum) + next[i] * momentum;
  }
}

export function batch_norm(input, runningMean, runningVar, weight, bias, training = true, eps = 1e-5, momentum = 0.1) {
  const symbolic = input instanceof SymbolicTensor || input.isSymbolic;
  if (!training) {
    return ops.batch_norm(input, weight, bias, runningMean, runningVar, CHANNEL_AXIS, eps);
  }

  const dims = [];
  for (let i = 0; i < input.ndim; i++) {
    if (i !== CHANNEL_AXIS) dims.push(i);
  }

  const batchMean = reduceMeanOver(input, dims);
  const centered = ops.sub(input, batchMean);
  const batchVar = reduceMeanOver(ops.mul(centered, centered), dims);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(batchVar, full([], eps))));
  let normalized = ops.mul(centered, invStd);

  const affineShape = channelShape(input.ndim, input.shape[CHANNEL_AXIS]);
  if (weight) normalized = ops.mul(normalized, weight.reshape(affineShape));
  if (bias) normalized = ops.add(normalized, bias.reshape(affineShape));

  if (!symbolic) {
    if (runningMean) blendRunning(runningMean, batchMean, momentum);
    if (runningVar) blendRunning(runningVar, batchVar, momentum);
  }
  return normalized;
}
