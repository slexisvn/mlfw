import * as ops from '../../tensor/ops/ops.js';
import type { NNTensor } from '../types.js';

export type UpsampleMode = 'nearest';

function normalizeScale(scale: number | readonly number[], spatialDims: number): number[] {
  const s = typeof scale === 'number' ? new Array(spatialDims).fill(scale) : [...scale];
  if (s.length !== spatialDims) {
    throw new Error(`interpolate: scale_factor has ${s.length} entr(ies) but the input has ${spatialDims} spatial dimension(s)`);
  }
  for (const v of s) {
    if (!Number.isInteger(v) || v < 1) {
      throw new Error(`interpolate: nearest mode needs integer scale factors >= 1, received ${JSON.stringify(scale)}`);
    }
  }
  return s;
}

function scalesFromSize(shape: readonly number[], size: readonly number[]): number[] {
  const spatial = shape.slice(2);
  if (size.length !== spatial.length) {
    throw new Error(`interpolate: size has ${size.length} entr(ies) but the input has ${spatial.length} spatial dimension(s)`);
  }
  return size.map((target, i) => {
    if (target % spatial[i] !== 0) {
      throw new Error(`interpolate: nearest mode can only upsample by an integer factor, but dimension ${i + 2} goes ${spatial[i]} -> ${target}`);
    }
    return target / spatial[i];
  });
}

export function interpolate(
  input: NNTensor,
  opts: { size?: readonly number[] | null; scaleFactor?: number | readonly number[] | null; mode?: UpsampleMode } = {},
): NNTensor {
  const { size = null, scaleFactor = null, mode = 'nearest' } = opts;
  if (mode !== 'nearest') throw new Error(`interpolate: unsupported mode '${mode}' (only 'nearest' is implemented)`);
  if ((size === null) === (scaleFactor === null)) {
    throw new Error('interpolate: pass exactly one of size or scaleFactor');
  }

  const shape = [...input.shape];
  const spatialDims = shape.length - 2;
  if (spatialDims < 1) throw new Error(`interpolate: expected an input of rank >= 3 (N, C, ...spatial) but got rank ${shape.length}`);

  const scales = size !== null ? scalesFromSize(shape, size) : normalizeScale(scaleFactor!, spatialDims);
  if (scales.every(s => s === 1)) return input;

  let current: NNTensor = input;
  let currentShape = shape;
  for (let d = 0; d < spatialDims; d++) {
    const s = scales[d];
    if (s === 1) continue;
    const axis = 2 + d;
    const expanded = [...currentShape.slice(0, axis + 1), 1, ...currentShape.slice(axis + 1)];
    const repeats = new Array(expanded.length).fill(1);
    repeats[axis + 1] = s;
    const merged = [...currentShape];
    merged[axis] = currentShape[axis] * s;
    current = ops.reshape(ops.repeat(ops.reshape(current, expanded), repeats), merged) as NNTensor;
    currentShape = merged;
  }
  return current;
}

export function upsample_nearest(input: NNTensor, scaleFactor: number | readonly number[]): NNTensor {
  return interpolate(input, { scaleFactor, mode: 'nearest' });
}
