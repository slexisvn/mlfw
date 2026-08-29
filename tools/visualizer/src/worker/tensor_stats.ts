import type { TensorPreview, TensorStats } from '../protocol.js';

const PREVIEW_VALUES = 8;

export type TensorLike = {
  shape?: readonly number[];
  dtype?: unknown;
  data?: ArrayLike<number>;
  contiguous?: () => { data: ArrayLike<number> };
};

export function valuesOf(tensor: TensorLike | null | undefined): number[] {
  if (!tensor) return [];
  const array = tensor.contiguous ? tensor.contiguous().data : tensor.data;
  if (!array) return [];
  const values = new Array<number>(array.length);
  for (let i = 0; i < array.length; i++) values[i] = Number(array[i]);
  return values;
}

export function statsOf(values: readonly number[]): TensorStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let squares = 0;
  let finite = 0;
  let zeros = 0;
  let nan = 0;
  let inf = 0;

  for (const value of values) {
    if (Number.isNaN(value)) { nan++; continue; }
    if (!Number.isFinite(value)) { inf++; continue; }
    if (value === 0) zeros++;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    squares += value * value;
    finite++;
  }

  const mean = finite === 0 ? null : sum / finite;
  return {
    min: finite === 0 ? null : min,
    max: finite === 0 ? null : max,
    mean,
    std: mean === null ? null : Math.sqrt(Math.max(0, squares / finite - mean * mean)),
    norm: Math.sqrt(squares),
    zeros,
    nan,
    inf,
  };
}

export function describe(name: string, tensor: TensorLike | null | undefined, values: readonly number[]): TensorPreview {
  return {
    name,
    shape: [...(tensor?.shape ?? [])],
    dtype: String(tensor?.dtype ?? 'f32'),
    numel: values.length,
    preview: values.slice(0, PREVIEW_VALUES),
    stats: statsOf(values),
  };
}

export function previewsOf(
  tensors: readonly (TensorLike | null | undefined)[],
  nameAt: (index: number) => string,
): { previews: TensorPreview[]; values: number[][] } {
  const values = tensors.map(valuesOf);
  return { previews: tensors.map((tensor, i) => describe(nameAt(i), tensor, values[i])), values };
}

export function unhealthy(previews: readonly TensorPreview[]): { nan: number; inf: number } {
  let nan = 0;
  let inf = 0;
  for (const preview of previews) {
    nan += preview.stats.nan;
    inf += preview.stats.inf;
  }
  return { nan, inf };
}
