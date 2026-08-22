import { nextPow2 } from '../../../util/numeric_array.js';
import * as ops from '../../ops/ops.js';
import { zeros, ones } from '../../factory/creation_ops.js';
import { tensor } from '../../factory/from_ops.js';
import { normalizeAxis as normDim } from '../../utils/shape_utils.js';
import type { Tensor } from '../../core/tensor.js';
import type { NativeKernel } from '../types.js';

type SortResult = { values: Tensor; indices: Tensor };

function asTensor(value: unknown): Tensor {
  return value as Tensor;
}

function asTensorArray(value: unknown): Tensor[] {
  return value as Tensor[];
}

export function scatterKernel(self: Tensor, dim: number, index: Tensor, src: Tensor): Tensor {
  const z = zeros(self.shape, { dtype: self.dtype });
  const scattered = asTensor(ops.scatter_add(z, dim, index, src));
  const counts = asTensor(ops.scatter_add(z, dim, index, ones(src.shape, { dtype: self.dtype })));
  return asTensor(ops.where(asTensor(ops.gt(counts, zeros(counts.shape, { dtype: counts.dtype }))), scattered, self));
}

export function repeatKernel(self: Tensor, reps: readonly number[]): Tensor {
  const shape = self.shape;
  const ndim = shape.length;
  if (reps.length < ndim) throw new Error('repeat: reps length must be >= tensor rank');
  const lead = reps.length - ndim;
  const aligned = lead > 0 ? [...Array(lead).fill(1), ...shape] : shape.slice();
  const interleaved: number[] = [];
  const expanded: number[] = [];
  const finalShape: number[] = [];
  for (let i = 0; i < aligned.length; i++) {
    interleaved.push(1, aligned[i]);
    expanded.push(reps[i], aligned[i]);
    finalShape.push(reps[i] * aligned[i]);
  }
  return asTensor(ops.reshape(asTensor(ops.expand(asTensor(ops.reshape(self, interleaved)), expanded)), finalShape));
}

export function tileKernel(self: Tensor, reps: readonly number[]): Tensor {
  const ndim = self.shape.length;
  const r = reps.length < ndim ? [...Array(ndim - reps.length).fill(1), ...reps] : reps;
  return asTensor(ops.repeat(self, r));
}

export function splitKernel(self: Tensor, sizes: readonly number[], dim: number): Tensor[] {
  const out: Tensor[] = [];
  const rank = self.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  let start = 0;
  for (const s of sizes) {
    out.push(asTensor(ops.narrow(self, d, start, s)));
    start += s;
  }
  return out;
}

export function chunkKernel(self: Tensor, chunks: number, dim: number): Tensor[] {
  const rank = self.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  const n = self.shape[d];
  const size = Math.ceil(n / chunks);
  return asTensorArray(ops.split(self, size, d));
}

export function rollKernel(self: Tensor, shift: number, dim = 0): Tensor {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  const n = self.shape[d];
  const s = ((shift % n) + n) % n;
  if (s === 0) return self;
  return asTensor(ops.cat([asTensor(ops.narrow(self, d, n - s, s)), asTensor(ops.narrow(self, d, 0, n - s))], d));
}

export function flipKernel(self: Tensor, dims: number | readonly number[]): Tensor {
  const dimList = Array.isArray(dims) ? dims : [dims];
  const rank = self.shape.length;
  let out = self;
  for (const dim of dimList) {
    const d = normDim(dim, rank);
    const n = out.shape[d];
    const rev = new Array<number>(n);
    for (let i = 0; i < n; i++) rev[i] = n - 1 - i;
    out = asTensor(ops.index_select(out, d, tensor(rev, { dtype: 'i32' })));
  }
  return out;
}

export function cumsumKernel(self: Tensor, dim = 0): Tensor {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  const n = self.shape[d];
  let out = self;
  for (let k = 1; k < n; k *= 2) {
    const zShape = [...out.shape];
    zShape[d] = k;
    const shifted = asTensor(ops.cat([zeros(zShape, { dtype: out.dtype }), asTensor(ops.narrow(out, d, 0, n - k))], d));
    out = asTensor(ops.add(out, shifted));
  }
  return out;
}

function bitonicLastDim(self: Tensor, descending: boolean, withIdx: false): Tensor;
function bitonicLastDim(self: Tensor, descending: boolean, withIdx: true): SortResult;
function bitonicLastDim(self: Tensor, descending: boolean, withIdx: boolean): Tensor | SortResult {
  const rank = self.shape.length;
  const lastDim = rank - 1;
  const n = self.shape[lastDim];
  const p = nextPow2(n);
  const sentinel = descending ? -Infinity : Infinity;

  let x = self;
  if (p > n) {
    const low = new Array(rank).fill(0);
    const high = new Array(rank).fill(0);
    high[lastDim] = p - n;
    x = asTensor(ops.pad(self, low, high, sentinel));
  }

  const bcastShape = new Array(rank).fill(1);
  bcastShape[lastDim] = p;

  let idx: Tensor | null = null;
  if (withIdx) {
    const iotaArr = new Array<number>(p);
    for (let i = 0; i < p; i++) iotaArr[i] = i;
    idx = asTensor(ops.reshape(tensor(iotaArr, { dtype: 'i32' }), bcastShape));
  }

  for (let k = 2; k <= p; k <<= 1) {
    for (let j = k >> 1; j >= 1; j >>= 1) {
      const partnerIdx = new Array<number>(p);
      const keepLo = new Array<number>(p);
      for (let i = 0; i < p; i++) {
        const ixj = i ^ j;
        partnerIdx[i] = ixj;
        const ascending = (i & k) === 0;
        const dir = descending ? !ascending : ascending;
        if (ixj > i) keepLo[i] = dir ? 1 : 0;
        else keepLo[i] = dir ? 0 : 1;
      }
      const pConst = tensor(partnerIdx, { dtype: 'i32' });
      const partner = asTensor(ops.index_select(x, lastDim, pConst));
      const lo = asTensor(ops.minimum(x, partner));
      const hi = asTensor(ops.maximum(x, partner));
      const mask = asTensor(ops.reshape(tensor(keepLo, { dtype: 'f32' }), bcastShape));
      if (withIdx) {
        const partnerIdxVals = asTensor(ops.index_select(idx as Tensor, lastDim, pConst));
        const takeSelf = asTensor(ops.where(mask, asTensor(ops.eq(lo, x)), asTensor(ops.eq(hi, x))));
        idx = asTensor(ops.where(takeSelf, idx as Tensor, partnerIdxVals));
      }
      x = asTensor(ops.where(mask, lo, hi));
    }
  }

  if (p > n) {
    x = asTensor(ops.narrow(x, lastDim, 0, n));
    if (withIdx) idx = asTensor(ops.narrow(idx as Tensor, lastDim, 0, n));
  }
  return withIdx ? { values: x, indices: idx as Tensor } : x;
}

export function sortKernel(self: Tensor, dim = -1, descending = false): Tensor {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  if (d === rank - 1) return bitonicLastDim(self, descending, false);
  const x = asTensor(ops.transpose(self, d, rank - 1));
  const sorted = bitonicLastDim(x, descending, false);
  return asTensor(ops.transpose(sorted, d, rank - 1));
}

function sortWithIndices(self: Tensor, dim: number, descending: boolean): SortResult {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  if (d === rank - 1) return bitonicLastDim(self, descending, true);
  const x = asTensor(ops.transpose(self, d, rank - 1));
  const r = bitonicLastDim(x, descending, true);
  return { values: asTensor(ops.transpose(r.values, d, rank - 1)), indices: asTensor(ops.transpose(r.indices, d, rank - 1)) };
}

export function argsortKernel(self: Tensor, dim = -1, descending = false): Tensor {
  return sortWithIndices(self, dim, descending).indices;
}

export function topkKernel(self: Tensor, k: number, dim = -1, largest = true): Tensor[] {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  const { values, indices } = sortWithIndices(self, d, largest);
  return [asTensor(ops.narrow(values, d, 0, k)), asTensor(ops.narrow(indices, d, 0, k))];
}

export const COMPOSITE_KERNELS: Readonly<Record<string, NativeKernel>> = Object.freeze({
  scatter: (_keySet, self, dim, index, src) => scatterKernel(self as Tensor, dim as number, index as Tensor, src as Tensor),
  repeat: (_keySet, self, reps) => repeatKernel(self as Tensor, reps as readonly number[]),
  tile: (_keySet, self, reps) => tileKernel(self as Tensor, reps as readonly number[]),
  split: (_keySet, self, sizes, dim) => splitKernel(self as Tensor, sizes as readonly number[], dim as number),
  chunk: (_keySet, self, chunks, dim) => chunkKernel(self as Tensor, chunks as number, dim as number),
  roll: (_keySet, self, shift, dim) => rollKernel(self as Tensor, shift as number, dim as number),
  flip: (_keySet, self, dims) => flipKernel(self as Tensor, dims as number | readonly number[]),
  cumsum: (_keySet, self, dim) => cumsumKernel(self as Tensor, dim as number),
  sort: (_keySet, self, dim, descending) => sortKernel(self as Tensor, dim as number, descending as boolean),
  argsort: (_keySet, self, dim, descending) => argsortKernel(self as Tensor, dim as number, descending as boolean),
  topk: (_keySet, self, k, dim, largest) => topkKernel(self as Tensor, k as number, dim as number, largest as boolean),
});
