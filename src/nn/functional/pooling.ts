import * as ops from '../../tensor/ops/ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export type Pair2 = [number, number];
export type PairPadding2d = [Pair2, Pair2] | Pair2;
export type Pool2dSize = number | Pair2;
export type Pool2dPadding = number | PairPadding2d;

export function max_pool2d(input: Tensor, kernelSize: Pool2dSize, stride?: Pool2dSize | null, padding: Pool2dPadding = [[0, 0], [0, 0]]): Tensor {
  const ks = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
  const s = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : ks;
  const p = _normalizePadding(padding);
  return ops.pool2d(input, 'max', ks, s, p);
}

export function avg_pool2d(input: Tensor, kernelSize: Pool2dSize, stride?: Pool2dSize | null, padding: Pool2dPadding = [[0, 0], [0, 0]]): Tensor {
  const ks = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
  const s = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : ks;
  const p = _normalizePadding(padding);
  return ops.pool2d(input, 'avg', ks, s, p);
}

export function adaptive_avg_pool2d(input: Tensor, outputSize: Pool2dSize): Tensor {
  const [outH, outW] = Array.isArray(outputSize) ? outputSize : [outputSize, outputSize];
  const inH = input.shape[2];
  const inW = input.shape[3];
  const kH = Math.floor(inH / outH);
  const kW = Math.floor(inW / outW);
  const sH = kH;
  const sW = kW;
  return avg_pool2d(input, [kH, kW], [sH, sW], [[0, 0], [0, 0]]);
}

function _normalizePadding(padding: Pool2dPadding): PairPadding2d {
  if (typeof padding === 'number') return [[padding, padding], [padding, padding]];
  if (Array.isArray(padding) && typeof padding[0] === 'number') {
    const pairs = padding.map(p => [p, p] as Pair2);
    if (pairs.length === 1) return [pairs[0], pairs[0]];
    return pairs as PairPadding2d;
  }
  return padding;
}
