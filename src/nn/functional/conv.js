import * as ops from '../../tensor/ops/ops.js';
import { reshape, unsqueeze, squeeze } from '../../tensor/view/view_ops.js';

export function conv2d(input, weight, bias, stride = [1, 1], padding = [[0, 0], [0, 0]], dilation = [1, 1], groups = 1) {
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

export function conv1d(input, weight, bias, stride = 1, padding = 0, dilation = 1, groups = 1) {
  const input4d = unsqueeze(input, 2);
  const weight4d = unsqueeze(weight, 2);
  const s = [1, Array.isArray(stride) ? stride[0] : stride];
  const p = [[0, 0], _normalizePadding1d(padding)];
  const d = [1, Array.isArray(dilation) ? dilation[0] : dilation];
  const out4d = conv2d(input4d, weight4d, null, s, p, d, groups);
  const out = squeeze(out4d, 2);
  if (bias) {
    const biasView = reshape(bias, [1, bias.shape[0], 1]);
    return ops.add(out, biasView);
  }
  return out;
}

function _normalizePadding(padding) {
  if (typeof padding === 'number') return [[padding, padding], [padding, padding]];
  if (Array.isArray(padding) && typeof padding[0] === 'number') {
    return padding.map(p => [p, p]);
  }
  return padding;
}

function _normalizePadding1d(padding) {
  if (typeof padding === 'number') return [padding, padding];
  return padding;
}
