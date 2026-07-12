import * as ops from '../../ops/ops.js';
import { zeros, ones } from '../../factory/creation_ops.js';
import { tensor } from '../../factory/from_ops.js';
import { normalizeAxis as normDim } from '../../utils/shape_utils.js';

export function scatterKernel(self, dim, index, src) {
  const z = zeros(self.shape, { dtype: self.dtype });
  const scattered = ops.scatter_add(z, dim, index, src);
  const counts = ops.scatter_add(z, dim, index, ones(src.shape, { dtype: self.dtype }));
  return ops.where(ops.gt(counts, zeros(counts.shape, { dtype: counts.dtype })), scattered, self);
}

export function repeatKernel(self, reps) {
  const shape = self.shape;
  const ndim = shape.length;
  if (reps.length < ndim) throw new Error('repeat: reps length must be >= tensor rank');
  const lead = reps.length - ndim;
  const aligned = lead > 0 ? [...Array(lead).fill(1), ...shape] : shape.slice();
  const interleaved = [];
  const expanded = [];
  const finalShape = [];
  for (let i = 0; i < aligned.length; i++) {
    interleaved.push(1, aligned[i]);
    expanded.push(reps[i], aligned[i]);
    finalShape.push(reps[i] * aligned[i]);
  }
  return ops.reshape(ops.expand(ops.reshape(self, interleaved), expanded), finalShape);
}

export function tileKernel(self, reps) {
  const ndim = self.shape.length;
  const r = reps.length < ndim ? [...Array(ndim - reps.length).fill(1), ...reps] : reps;
  return ops.repeat(self, r);
}

export function splitKernel(self, sizes, dim) {
  const out = [];
  const rank = self.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  let start = 0;
  for (const s of sizes) {
    out.push(ops.narrow(self, d, start, s));
    start += s;
  }
  return out;
}

export function chunkKernel(self, chunks, dim) {
  const rank = self.shape.length;
  const d = dim < 0 ? rank + dim : dim;
  const n = self.shape[d];
  const size = Math.ceil(n / chunks);
  return ops.split(self, size, d);
}

export function rollKernel(self, shift, dim = 0) {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  const n = self.shape[d];
  const s = ((shift % n) + n) % n;
  if (s === 0) return self;
  return ops.cat([ops.narrow(self, d, n - s, s), ops.narrow(self, d, 0, n - s)], d);
}

export function flipKernel(self, dims) {
  const dimList = Array.isArray(dims) ? dims : [dims];
  const rank = self.shape.length;
  let out = self;
  for (const dim of dimList) {
    const d = normDim(dim, rank);
    const n = out.shape[d];
    const rev = new Array(n);
    for (let i = 0; i < n; i++) rev[i] = n - 1 - i;
    out = ops.index_select(out, d, tensor(rev, { dtype: 'i32' }));
  }
  return out;
}

export function cumsumKernel(self, dim = 0) {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  const n = self.shape[d];
  let out = self;
  for (let k = 1; k < n; k *= 2) {
    const zShape = [...out.shape];
    zShape[d] = k;
    const shifted = ops.cat([zeros(zShape, { dtype: out.dtype }), ops.narrow(out, d, 0, n - k)], d);
    out = ops.add(out, shifted);
  }
  return out;
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function bitonicLastDim(self, descending, withIdx) {
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
    x = ops.pad(self, low, high, sentinel);
  }

  const bcastShape = new Array(rank).fill(1);
  bcastShape[lastDim] = p;

  let idx = null;
  if (withIdx) {
    const iotaArr = new Array(p);
    for (let i = 0; i < p; i++) iotaArr[i] = i;
    idx = ops.reshape(tensor(iotaArr, { dtype: 'i32' }), bcastShape);
  }

  for (let k = 2; k <= p; k <<= 1) {
    for (let j = k >> 1; j >= 1; j >>= 1) {
      const partnerIdx = new Array(p);
      const keepLo = new Array(p);
      for (let i = 0; i < p; i++) {
        const ixj = i ^ j;
        partnerIdx[i] = ixj;
        const ascending = (i & k) === 0;
        const dir = descending ? !ascending : ascending;
        if (ixj > i) keepLo[i] = dir ? 1 : 0;
        else keepLo[i] = dir ? 0 : 1;
      }
      const pConst = tensor(partnerIdx, { dtype: 'i32' });
      const partner = ops.index_select(x, lastDim, pConst);
      const lo = ops.minimum(x, partner);
      const hi = ops.maximum(x, partner);
      const mask = ops.reshape(tensor(keepLo, { dtype: 'f32' }), bcastShape);
      if (withIdx) {
        const partnerIdxVals = ops.index_select(idx, lastDim, pConst);
        const takeSelf = ops.where(mask, ops.eq(lo, x), ops.eq(hi, x));
        idx = ops.where(takeSelf, idx, partnerIdxVals);
      }
      x = ops.where(mask, lo, hi);
    }
  }

  if (p > n) {
    x = ops.narrow(x, lastDim, 0, n);
    if (withIdx) idx = ops.narrow(idx, lastDim, 0, n);
  }
  return withIdx ? { values: x, indices: idx } : x;
}

export function sortKernel(self, dim = -1, descending = false) {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  if (d === rank - 1) return bitonicLastDim(self, descending, false);
  const x = ops.transpose(self, d, rank - 1);
  const sorted = bitonicLastDim(x, descending, false);
  return ops.transpose(sorted, d, rank - 1);
}

function sortWithIndices(self, dim, descending) {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  if (d === rank - 1) return bitonicLastDim(self, descending, true);
  const x = ops.transpose(self, d, rank - 1);
  const r = bitonicLastDim(x, descending, true);
  return { values: ops.transpose(r.values, d, rank - 1), indices: ops.transpose(r.indices, d, rank - 1) };
}

export function argsortKernel(self, dim = -1, descending = false) {
  return sortWithIndices(self, dim, descending).indices;
}

export function topkKernel(self, k, dim = -1, largest = true) {
  const rank = self.shape.length;
  const d = normDim(dim, rank);
  const { values, indices } = sortWithIndices(self, d, largest);
  return [ops.narrow(values, d, 0, k), ops.narrow(indices, d, 0, k)];
}

export const COMPOSITE_KERNELS = Object.freeze({
  scatter: (keySet, self, dim, index, src) => scatterKernel(self, dim, index, src),
  repeat: (keySet, self, reps) => repeatKernel(self, reps),
  tile: (keySet, self, reps) => tileKernel(self, reps),
  split: (keySet, self, sizes, dim) => splitKernel(self, sizes, dim),
  chunk: (keySet, self, chunks, dim) => chunkKernel(self, chunks, dim),
  roll: (keySet, self, shift, dim) => rollKernel(self, shift, dim),
  flip: (keySet, self, dims) => flipKernel(self, dims),
  cumsum: (keySet, self, dim) => cumsumKernel(self, dim),
  sort: (keySet, self, dim, descending) => sortKernel(self, dim, descending),
  argsort: (keySet, self, dim, descending) => argsortKernel(self, dim, descending),
  topk: (keySet, self, k, dim, largest) => topkKernel(self, k, dim, largest),
});
