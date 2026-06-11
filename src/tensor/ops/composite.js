import { cat, add, index_select, minimum, maximum, where, pad } from './ops.js';
import { zeros } from '../factory/creation_ops.js';
import { tensor } from '../factory/from_ops.js';

function _normDim(d, rank) {
  return d < 0 ? rank + d : d;
}

export function roll(self, shift, dim = 0) {
  const rank = self.shape.length;
  const d = _normDim(dim, rank);
  const n = self.shape[d];
  const s = ((shift % n) + n) % n;
  if (s === 0) return self;
  return cat([self.narrow(d, n - s, s), self.narrow(d, 0, n - s)], d);
}

export function flip(self, dims) {
  const dimList = Array.isArray(dims) ? dims : [dims];
  const rank = self.shape.length;
  let out = self;
  for (const dim of dimList) {
    const d = _normDim(dim, rank);
    const n = out.shape[d];
    const rev = new Array(n);
    for (let i = 0; i < n; i++) rev[i] = n - 1 - i;
    out = index_select(out, d, tensor(rev, { dtype: 'i32' }));
  }
  return out;
}

export function cumsum(self, dim = 0) {
  const rank = self.shape.length;
  const d = _normDim(dim, rank);
  const n = self.shape[d];
  let out = self;
  for (let k = 1; k < n; k *= 2) {
    const zShape = [...out.shape];
    zShape[d] = k;
    const shifted = cat([zeros(zShape, { dtype: out.dtype }), out.narrow(d, 0, n - k)], d);
    out = add(out, shifted);
  }
  return out;
}

function _nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function _bitonicLastDim(self, descending) {
  const rank = self.shape.length;
  const lastDim = rank - 1;
  const n = self.shape[lastDim];
  const P = _nextPow2(n);
  const sentinel = descending ? -Infinity : Infinity;

  let x = self;
  if (P > n) {
    const low = new Array(rank).fill(0);
    const high = new Array(rank).fill(0);
    high[lastDim] = P - n;
    x = pad(self, low, high, sentinel);
  }

  const bcastShape = new Array(rank).fill(1);
  bcastShape[lastDim] = P;

  for (let k = 2; k <= P; k <<= 1) {
    for (let j = k >> 1; j >= 1; j >>= 1) {
      const partnerIdx = new Array(P);
      const keepLo = new Array(P);
      for (let i = 0; i < P; i++) {
        const ixj = i ^ j;
        partnerIdx[i] = ixj;
        const ascending = (i & k) === 0;
        const dir = descending ? !ascending : ascending;
        if (ixj > i) keepLo[i] = dir ? 1 : 0;
        else keepLo[i] = dir ? 0 : 1;
      }
      const partner = index_select(x, lastDim, tensor(partnerIdx, { dtype: 'i32' }));
      const lo = minimum(x, partner);
      const hi = maximum(x, partner);
      const mask = tensor(keepLo, { dtype: 'f32' }).reshape(bcastShape);
      x = where(mask, lo, hi);
    }
  }

  if (P > n) x = x.narrow(lastDim, 0, n);
  return x;
}

export function sort(self, dim = -1, descending = false) {
  const rank = self.shape.length;
  const d = _normDim(dim, rank);
  if (d === rank - 1) return _bitonicLastDim(self, descending);
  const x = self.transpose(d, rank - 1);
  const sorted = _bitonicLastDim(x, descending);
  return sorted.transpose(d, rank - 1);
}

export function topk(self, k, dim = -1, largest = true) {
  const rank = self.shape.length;
  const d = _normDim(dim, rank);
  const sorted = sort(self, d, largest);
  return sorted.narrow(d, 0, k);
}
