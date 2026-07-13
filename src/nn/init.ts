import type { Tensor } from '../tensor/core/tensor.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';

type KaimingMode = 'fan_in' | 'fan_out';
type NumericFillArray = Exclude<NumericTypedArray, BigInt64Array>;

export function _calculateFanInFanOut(tensor: Tensor): { fanIn: number; fanOut: number } {
  const shape = tensor.shape;
  const ndim = shape.length;
  if (ndim < 2) throw new Error('Fan in/out requires at least 2D tensor');
  const numInputFmaps = shape[1];
  const numOutputFmaps = shape[0];
  let receptiveFieldSize = 1;
  for (let i = 2; i < ndim; i++) receptiveFieldSize *= shape[i];
  return {
    fanIn: numInputFmaps * receptiveFieldSize,
    fanOut: numOutputFmaps * receptiveFieldSize,
  };
}

export function resetLinearParameters(weight: Tensor, bias?: Tensor | null): void {
  kaiming_uniform_(weight, Math.sqrt(5));
  if (bias) {
    const { fanIn } = _calculateFanInFanOut(weight);
    const bound = 1 / Math.sqrt(fanIn);
    uniform_(bias, -bound, bound);
  }
}

export function uniform_<T extends Tensor>(tensor: T, a = 0, b = 1): T {
  const data = tensor._impl.storage.data;
  if (!data) return tensor;
  const range = b - a;
  for (let i = 0; i < data.length; i++) data[i] = a + Math.random() * range;
  tensor._impl.bumpVersion();
  return tensor;
}

export function normal_<T extends Tensor>(tensor: T, mean = 0, std = 1): T {
  const data = tensor._impl.storage.data;
  if (!data) return tensor;
  for (let i = 0; i < data.length; i += 2) {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 6.283185307179586 * u2;
    data[i] = mean + std * r * Math.cos(theta);
    if (i + 1 < data.length) data[i + 1] = mean + std * r * Math.sin(theta);
  }
  tensor._impl.bumpVersion();
  return tensor;
}

export function zeros_<T extends Tensor>(tensor: T): T {
  const data = tensor._impl.storage.data;
  if (data) (data as NumericFillArray).fill(0);
  tensor._impl.bumpVersion();
  return tensor;
}

export function ones_<T extends Tensor>(tensor: T): T {
  const data = tensor._impl.storage.data;
  if (data) (data as NumericFillArray).fill(1);
  tensor._impl.bumpVersion();
  return tensor;
}

export function constant_<T extends Tensor>(tensor: T, val: number): T {
  const data = tensor._impl.storage.data;
  if (data) (data as NumericFillArray).fill(val);
  tensor._impl.bumpVersion();
  return tensor;
}

export function xavier_uniform_<T extends Tensor>(tensor: T, gain = 1.0): T {
  const { fanIn, fanOut } = _calculateFanInFanOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  const a = Math.sqrt(3.0) * std;
  return uniform_(tensor, -a, a);
}

export function xavier_normal_<T extends Tensor>(tensor: T, gain = 1.0): T {
  const { fanIn, fanOut } = _calculateFanInFanOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  return normal_(tensor, 0, std);
}

export function kaiming_uniform_<T extends Tensor>(tensor: T, a = 0, mode: KaimingMode = 'fan_in', nonlinearity = 'leaky_relu'): T {
  const { fanIn, fanOut } = _calculateFanInFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  const gain = _calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  const bound = Math.sqrt(3.0) * std;
  return uniform_(tensor, -bound, bound);
}

export function kaiming_normal_<T extends Tensor>(tensor: T, a = 0, mode: KaimingMode = 'fan_in', nonlinearity = 'leaky_relu'): T {
  const { fanIn, fanOut } = _calculateFanInFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  const gain = _calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  return normal_(tensor, 0, std);
}

function _calculateGain(nonlinearity: string, param = 0.01): number {
  switch (nonlinearity) {
    case 'linear': case 'sigmoid': return 1;
    case 'tanh': return 5.0 / 3;
    case 'relu': return Math.sqrt(2.0);
    case 'leaky_relu': return Math.sqrt(2.0 / (1 + param * param));
    default: return 1;
  }
}
