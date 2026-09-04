import { TensorType, DYNAMIC } from './types.js';
import type { Dim } from './types.js';
import type { Value } from './value.js';

type DimFilter = (dim: Dim | null | undefined) => boolean;

const KNOWN_EXTENT: DimFilter = (dim) => dim !== DYNAMIC && dim !== null && dim !== undefined;

const SYMBOL_NAME: DimFilter = (dim) => typeof dim === 'string';

export function symbolicDimsOf(value: Value): Dim[] | null {
  const symbols = value.symbolicShape;
  return symbols ? [...symbols] : null;
}

export function carriesSymbol(dims: readonly Dim[] | null): boolean {
  return dims !== null && dims.some(SYMBOL_NAME);
}

function merge(into: Value, from: readonly Dim[], known: DimFilter): boolean {
  const type = into.type;
  if (!(type instanceof TensorType) || type.rank !== from.length) return false;
  const dims = symbolicDimsOf(into) ?? [...type.shape];
  let learned = false;
  for (let axis = 0; axis < type.rank; axis++) {
    if (type.shape[axis] !== DYNAMIC) continue;
    if (known(dims[axis])) continue;
    if (!known(from[axis])) continue;
    dims[axis] = from[axis];
    learned = true;
  }
  if (learned || into.symbolicShape === undefined) into.symbolicShape = dims;
  return learned;
}

export function learnDynamicExtents(into: Value, from: readonly Dim[]): boolean {
  return merge(into, from, KNOWN_EXTENT);
}

export function learnSymbolNames(into: Value, from: readonly Dim[]): boolean {
  return merge(into, from, SYMBOL_NAME);
}
