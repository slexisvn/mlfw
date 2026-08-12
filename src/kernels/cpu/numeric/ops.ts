import { hostMatrix, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { tensorToContiguous } from '../../../dispatcher/jit_dispatch.js';
import { qrHost } from './qr.js';
import { fftHost } from './fft.js';
import type { Tensor } from '../../../tensor/core/tensor.js';

type ComplexSignal = { re: Float64Array; im: Float64Array; n: number };

export function cpuQr(_keySet: unknown, a: Tensor): Tensor[] {
  const { data, rows, cols } = hostMatrix(a);
  const { Q, R, k } = qrHost(data, rows, cols);
  return [
    toHostTensor(Q, [rows, k], a.dtype, a.device),
    toHostTensor(R, [k, cols], a.dtype, a.device),
  ];
}

function splitComplex(x: Tensor): ComplexSignal {
  const data = tensorToContiguous(x) as ArrayLike<number>;
  if (x.ndim === 1) {
    const n = x.shape[0];
    return { re: Float64Array.from(data), im: new Float64Array(n), n };
  }
  if (x.ndim === 2 && x.shape[1] === 2) {
    const n = x.shape[0];
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = data[2 * i];
      im[i] = data[2 * i + 1];
    }
    return { re, im, n };
  }
  throw new Error(`fft: expected a 1-D real signal or [n, 2] complex tensor, got shape [${x.shape}]`);
}

function interleave(re: Float64Array, im: Float64Array, n: number): Float64Array {
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    out[2 * i] = re[i];
    out[2 * i + 1] = im[i];
  }
  return out;
}

function transform(x: Tensor, invert: boolean): Tensor {
  const { re, im, n } = splitComplex(x);
  fftHost(re, im, invert);
  return toHostTensor(interleave(re, im, n), [n, 2], x.dtype, x.device);
}

export function cpuFft(_keySet: unknown, x: Tensor): Tensor {
  return transform(x, false);
}

export function cpuIfft(_keySet: unknown, x: Tensor): Tensor {
  return transform(x, true);
}
