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
