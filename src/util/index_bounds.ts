export type IndexedTensor = {
  shape: readonly number[];
  strides?: readonly number[];
  _impl?: {
    storage?: { data?: ArrayLike<number | bigint> | null; isMeta?: boolean } | null;
    storageOffset?: number;
  } | null;
};

export type ArgIndexBound = { argIndex: number; limit: number; opName: string };

function storageOf(t: IndexedTensor): ArrayLike<number | bigint> | null {
  const storage = t && t._impl ? t._impl.storage : null;
  if (!storage || storage.isMeta) return null;
  const data = storage.data;
  return data && data.length > 0 ? data : null;
}

function numelOf(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function reportOutOfRange(value: number, position: number, limit: number, label: string): never {
  throw new RangeError(
    `${label}: index ${value} at position ${position} is out of range for a table of ${limit} row(s); valid indices are 0..${limit - 1}`
  );
}

export function assertIndicesInRange(indices: IndexedTensor, limit: number, label: string): void {
  const data = storageOf(indices);
  if (data === null || !Number.isFinite(limit) || limit <= 0) return;

  const shape = indices.shape;
  const n = numelOf(shape);
  if (n === 0) return;

  const offset = indices._impl?.storageOffset ?? 0;
  const strides = indices.strides ?? null;

  if (!strides || (offset === 0 && data.length === n)) {
    const end = Math.min(data.length, offset + n);
    for (let i = offset; i < end; i++) {
      const v = Number(data[i]);
      if (v < 0 || v >= limit) reportOutOfRange(v, i - offset, limit, label);
    }
    return;
  }

  const rank = shape.length;
  const counter = new Array<number>(rank).fill(0);
  let flat = offset;
  for (let pos = 0; pos < n; pos++) {
    if (flat >= 0 && flat < data.length) {
      const v = Number(data[flat]);
      if (v < 0 || v >= limit) reportOutOfRange(v, pos, limit, label);
    }
    for (let d = rank - 1; d >= 0; d--) {
      counter[d]++;
      flat += strides[d];
      if (counter[d] < shape[d]) break;
      flat -= strides[d] * shape[d];
      counter[d] = 0;
    }
  }
}

export function assertArgIndexBounds(bounds: readonly ArgIndexBound[], inputs: readonly IndexedTensor[]): void {
  for (const b of bounds) {
    const t = inputs[b.argIndex];
    if (t) assertIndicesInRange(t, b.limit, `${b.opName}: compiled input ${b.argIndex}`);
  }
}
