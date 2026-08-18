import * as ops from '../../tensor/ops/ops.js';
import { reshape, unsqueeze, squeeze } from '../../tensor/ops/ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export type Pair2 = [number, number];
export type PairPadding2d = [Pair2, Pair2] | Pair2;
export type ConvSize2d = number | Pair2;
export type ConvPadding2d = number | PairPadding2d;

export function conv2d(input: Tensor, weight: Tensor, bias: Tensor | null, stride: ConvSize2d = [1, 1], padding: ConvPadding2d = [[0, 0], [0, 0]], dilation: ConvSize2d = [1, 1], groups = 1): Tensor {
  const s = Array.isArray(stride) ? stride : [stride, stride];
  const p = _normalizePadding(padding);
  const d = Array.isArray(dilation) ? dilation : [dilation, dilation];

  const output = ops.conv2d(input, weight, s, p, d, groups);
  if (bias) {
    const biasView = reshape(bias, [1, bias.shape[0], 1, 1]);
    return ops.add(output, biasView);
  }
  return output;
}

export function conv1d(input: Tensor, weight: Tensor, bias: Tensor | null, stride: number | readonly number[] = 1, padding: number | Pair2 = 0, dilation: number | readonly number[] = 1, groups = 1): Tensor {
  const input4d = unsqueeze(input, 2);
  const weight4d = unsqueeze(weight, 2);
  const s = [1, Array.isArray(stride) ? (stride as readonly number[])[0] : stride] as Pair2;
  const p = [[0, 0], _normalizePadding1d(padding)] as [Pair2, Pair2];
  const d = [1, Array.isArray(dilation) ? (dilation as readonly number[])[0] : dilation] as Pair2;
  const out4d = conv2d(input4d, weight4d, null, s, p, d, groups);
  const out = squeeze(out4d, 2);
  if (bias) {
    const biasView = reshape(bias, [1, bias.shape[0], 1]);
    return ops.add(out, biasView);
  }
  return out;
}

export type Triple3 = [number, number, number];
export type ConvSize3d = number | Triple3;
export type ConvPadding3d = number | Triple3 | [Pair2, Pair2, Pair2];

function _triple(v: ConvSize3d): Triple3 {
  return typeof v === 'number' ? [v, v, v] : [...v] as Triple3;
}

function _normalizePadding3d(padding: ConvPadding3d): [Pair2, Pair2, Pair2] {
  if (typeof padding === 'number') return [[padding, padding], [padding, padding], [padding, padding]];
  if (typeof padding[0] === 'number') return (padding as Triple3).map(p => [p, p] as Pair2) as [Pair2, Pair2, Pair2];
  return padding as [Pair2, Pair2, Pair2];
}

export function conv3d(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
  stride: ConvSize3d = 1,
  padding: ConvPadding3d = 0,
  dilation: ConvSize3d = 1,
  groups = 1,
): Tensor {
  const s = _triple(stride), d = _triple(dilation);
  const p: readonly Pair2[] = _normalizePadding3d(padding);
  const [N, Ci, D] = input.shape;
  const [Co, , kD, kH, kW] = weight.shape;

  const paddedD = D + p[0][0] + p[0][1];
  const outD = Math.floor((paddedD - d[0] * (kD - 1) - 1) / s[0]) + 1;
  if (outD <= 0) throw new Error(`conv3d: depth ${D} with padding ${p[0]}, dilation ${d[0]} and kernel ${kD} produces a non-positive output depth`);

  const rank = input.shape.length;
  const low = new Array(rank).fill(0), high = new Array(rank).fill(0);
  low[2] = p[0][0];
  high[2] = p[0][1];
  const padded = (p[0][0] || p[0][1]) ? ops.pad(input, low, high) : input;

  const spatialPad: [Pair2, Pair2] = [p[1], p[2]];
  let acc: Tensor | null = null;
  for (let kd = 0; kd < kD; kd++) {
    const start = kd * d[0];
    const depthSlice = ops.slice(padded, 2, start, start + (outD - 1) * s[0] + 1, s[0]);
    const asBatch = reshape(ops.transpose(depthSlice, 1, 2), [N * outD, Ci, depthSlice.shape[3], depthSlice.shape[4]]);
    const planeWeight = reshape(ops.slice(weight, 2, kd, kd + 1), [Co, weight.shape[1], kH, kW]);
    const conv = conv2d(asBatch, planeWeight, null, [s[1], s[2]], spatialPad, [d[1], d[2]], groups);
    const back = ops.transpose(reshape(conv, [N, outD, Co, conv.shape[2], conv.shape[3]]), 1, 2);
    acc = acc === null ? back : ops.add(acc, back);
  }

  const out = acc as Tensor;
  if (bias) return ops.add(out, reshape(bias, [1, Co, 1, 1, 1]));
  return out;
}

function _dilateSpatial(input: Tensor, strides: Pair2): Tensor {
  let current = input;
  for (let d = 0; d < 2; d++) {
    const s = strides[d];
    if (s === 1) continue;
    const axis = 2 + d;
    const shape = [...current.shape];
    const expanded = [...shape.slice(0, axis + 1), 1, ...shape.slice(axis + 1)];
    const low = new Array(expanded.length).fill(0);
    const high = new Array(expanded.length).fill(0);
    high[axis + 1] = s - 1;
    const merged = [...shape];
    merged[axis] = shape[axis] * s;
    const padded = reshape(ops.pad(reshape(current, expanded), low, high), merged);
    current = ops.slice(padded, axis, 0, (shape[axis] - 1) * s + 1);
  }
  return current;
}

function _transposeConvWeight(weight: Tensor, groups: number): Tensor {
  const [cIn, cOutPerGroup, kH, kW] = weight.shape;
  const grouped = reshape(weight, [groups, cIn / groups, cOutPerGroup, kH, kW]);
  const swapped = ops.transpose(grouped, 1, 2);
  return ops.flip(reshape(swapped, [groups * cOutPerGroup, cIn / groups, kH, kW]), [2, 3]);
}

export function conv_transpose2d(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
  stride: ConvSize2d = [1, 1],
  padding: ConvPadding2d = [[0, 0], [0, 0]],
  outputPadding: ConvSize2d = [0, 0],
  dilation: ConvSize2d = [1, 1],
  groups = 1,
): Tensor {
  const s = (Array.isArray(stride) ? stride : [stride, stride]) as Pair2;
  const p = _normalizePadding(padding) as readonly Pair2[];
  const d = (Array.isArray(dilation) ? dilation : [dilation, dilation]) as Pair2;
  const op = (Array.isArray(outputPadding) ? outputPadding : [outputPadding, outputPadding]) as Pair2;

  const kH = weight.shape[2], kW = weight.shape[3];
  const equivalentPad: [Pair2, Pair2] = [
    [d[0] * (kH - 1) - p[0][0], d[0] * (kH - 1) - p[0][1] + op[0]],
    [d[1] * (kW - 1) - p[1][0], d[1] * (kW - 1) - p[1][1] + op[1]],
  ];
  for (const pair of equivalentPad) {
    for (const v of pair) {
      if (v < 0) throw new Error(`conv_transpose2d: padding ${JSON.stringify(padding)} exceeds dilation*(kernel-1); the output would be cropped below zero size`);
    }
  }

  const dilated = _dilateSpatial(input, s);
  const out = conv2d(dilated, _transposeConvWeight(weight, groups), null, [1, 1], equivalentPad, d, groups);

  if (bias) return ops.add(out, reshape(bias, [1, bias.shape[0], 1, 1]));
  return out;
}

export function conv_transpose1d(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
  stride: number | readonly number[] = 1,
  padding: number | Pair2 = 0,
  outputPadding: number | readonly number[] = 0,
  dilation: number | readonly number[] = 1,
  groups = 1,
): Tensor {
  const pick = (v: number | readonly number[]) => (Array.isArray(v) ? (v as readonly number[])[0] : v as number);
  const out4d = conv_transpose2d(
    unsqueeze(input, 2),
    unsqueeze(weight, 2),
    null,
    [1, pick(stride)],
    [[0, 0], _normalizePadding1d(padding)],
    [0, pick(outputPadding)],
    [1, pick(dilation)],
    groups,
  );
  const out = squeeze(out4d, 2);
  if (bias) return ops.add(out, reshape(bias, [1, bias.shape[0], 1]));
  return out;
}

function _normalizePadding(padding: ConvPadding2d): PairPadding2d {
  if (typeof padding === 'number') return [[padding, padding], [padding, padding]];
  if (Array.isArray(padding) && typeof padding[0] === 'number') {
    return padding.map(p => [p, p] as Pair2) as PairPadding2d;
  }
  return padding;
}

function _normalizePadding1d(padding: number | Pair2): Pair2 {
  if (typeof padding === 'number') return [padding, padding];
  return padding;
}
