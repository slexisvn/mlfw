import * as ops from '../../tensor/ops/ops.js';

export function max_pool2d(input, kernelSize, stride, padding = [[0, 0], [0, 0]]) {
  const ks = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
  const s = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : ks;
  const p = _normalizePadding(padding);
  return ops.pool2d(input, 'max', ks, s, p);
}

export function avg_pool2d(input, kernelSize, stride, padding = [[0, 0], [0, 0]]) {
  const ks = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
  const s = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : ks;
  const p = _normalizePadding(padding);
  return ops.pool2d(input, 'avg', ks, s, p);
}

export function adaptive_avg_pool2d(input, outputSize) {
  const [outH, outW] = Array.isArray(outputSize) ? outputSize : [outputSize, outputSize];
  const inH = input.shape[2];
  const inW = input.shape[3];
  const kH = Math.floor(inH / outH);
  const kW = Math.floor(inW / outW);
  const sH = kH;
  const sW = kW;
  return avg_pool2d(input, [kH, kW], [sH, sW], [[0, 0], [0, 0]]);
}

function _normalizePadding(padding) {
  if (typeof padding === 'number') return [[padding, padding], [padding, padding]];
  if (Array.isArray(padding) && typeof padding[0] === 'number') {
    const pairs = padding.map(p => [p, p]);
    if (pairs.length === 1) return [pairs[0], pairs[0]];
    return pairs;
  }
  return padding;
}
