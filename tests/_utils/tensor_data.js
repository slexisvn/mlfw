import { tensor } from '../../src/index.js';
import { isDtypeFloat } from '../../src/util/dtype_map.js';

export const numel = (shape) => shape.reduce((a, b) => a * b, 1);

export function nest(flat, shape) {
  if (shape.length === 1) return flat.slice(0, shape[0]);
  const sub = numel(shape.slice(1));
  const out = [];
  for (let i = 0; i < shape[0]; i++) out.push(nest(flat.slice(i * sub, (i + 1) * sub), shape.slice(1)));
  return out;
}

export function randomNested(rng, shape, lo = -1, hi = 1, dtype) {
  const n = numel(shape);
  const f = [];
  for (let i = 0; i < n; i++) {
    const v = lo + (hi - lo) * rng();
    f.push(dtype && !isDtypeFloat(dtype) ? Math.round(v) : v);
  }
  return nest(f, shape);
}

export const randomTensor = (rng, shape, lo = -1, hi = 1, dtype) =>
  tensor(randomNested(rng, shape, lo, hi, dtype), dtype ? { dtype } : undefined);

export function flat(v) {
  if (v && typeof v.contiguous === 'function') return Array.from(v.contiguous().data);
  if (v && v.data) return Array.from(v.data);
  if (Array.isArray(v)) return v.flat(Infinity);
  return [v];
}
