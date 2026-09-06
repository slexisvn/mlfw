import { SymInt } from '../sym_int.js';
import type { SymExpr } from '../sym_int.js';

export const AxeAxis = Object.freeze({
  MEM: 'm',
  LANE: 'lane',
  WARP: 'warp',
  REG: 'reg',
  BLOCK_X: 'block.x',
  BLOCK_Y: 'block.y',
  BLOCK_Z: 'block.z',
  THREAD_X: 'thread.x',
  THREAD_Y: 'thread.y',
  THREAD_Z: 'thread.z'
});

export type AxeAxisValue = (typeof AxeAxis)[keyof typeof AxeAxis];
export type AxeAxisName = string;
export type Iter = Readonly<{ extent: SymExpr; stride: SymExpr; axis: AxeAxisName }>;
export type Coord = ReadonlyMap<AxeAxisName, SymExpr>;
export type LayoutProver = Readonly<{ canProveEqual(a: SymExpr, b: SymExpr): boolean }>;
export type SliceRegion = Readonly<{ start: number; extent: number }>;

const EMPTY_COORD: Coord = new Map();
const DIFFERENTIAL_DOMAIN_LIMIT = 1 << 16;

export function iter(extent: SymExpr, stride: SymExpr, axis: AxeAxisName = AxeAxis.MEM): Iter {
  return Object.freeze({ extent, stride, axis });
}

export function coord(entries: Readonly<Record<AxeAxisName, SymExpr>> = {}): Coord {
  return new Map(Object.entries(entries));
}

function isPositiveConst(x: SymExpr): boolean {
  return typeof x === 'number' && Number.isInteger(x) && x > 0;
}

function isNonZeroConst(x: SymExpr): boolean {
  return typeof x === 'number' && Number.isInteger(x) && x !== 0;
}

function proveEqual(a: SymExpr, b: SymExpr, prover: LayoutProver | null): boolean {
  if (SymInt.equals(a, b)) return true;
  return prover ? prover.canProveEqual(a, b) : false;
}

function coordAdd(a: Coord, b: Coord): Coord {
  if (b.size === 0) return a;
  if (a.size === 0) return b;
  const out = new Map(a);
  for (const [axis, value] of b) out.set(axis, SymInt.add(out.get(axis) ?? 0, value));
  return out;
}

function coordAddTerm(target: Map<AxeAxisName, SymExpr>, axis: AxeAxisName, value: SymExpr): void {
  target.set(axis, SymInt.add(target.get(axis) ?? 0, value));
}

function coordEquals(a: Coord, b: Coord, prover: LayoutProver | null): boolean {
  const axes = new Set([...a.keys(), ...b.keys()]);
  for (const axis of axes) {
    if (!proveEqual(a.get(axis) ?? 0, b.get(axis) ?? 0, prover)) return false;
  }
  return true;
}

export function coordKey(c: Coord): string {
  return [...c.entries()]
    .filter(([, value]) => value !== 0)
    .sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0))
    .map(([axis, value]) => `${axis}=${value}`)
    .join(',');
}

function product(iters: readonly Iter[]): SymExpr {
  let out: SymExpr = 1;
  for (const it of iters) out = SymInt.mul(out, it.extent);
  return out;
}

function splitIter(it: Iter, outerExtent: number, innerExtent: number): readonly [Iter, Iter] {
  return [iter(outerExtent, SymInt.mul(innerExtent, it.stride), it.axis), iter(innerExtent, it.stride, it.axis)];
}

function normalizeShard(shard: readonly Iter[], prover: LayoutProver | null): Iter[] {
  const kept = shard.filter(it => !proveEqual(it.extent, 1, prover));
  const out: Iter[] = [];
  for (const it of kept) {
    const prev = out[out.length - 1];
    if (prev && prev.axis === it.axis && proveEqual(prev.stride, SymInt.mul(it.extent, it.stride), prover)) {
      out[out.length - 1] = iter(SymInt.mul(prev.extent, it.extent), it.stride, it.axis);
      continue;
    }
    out.push(it);
  }
  return out;
}

function normalizeReplica(
  replica: readonly Iter[],
  offset: Coord,
  prover: LayoutProver | null
): { replica: Iter[]; offset: Coord } {
  const shifted = new Map(offset);
  const kept: Iter[] = [];
  for (const it of replica) {
    if (proveEqual(it.extent, 1, prover)) continue;
    const stride = constOfStride(it.stride);
    if (stride !== null && stride < 0) {
      coordAddTerm(shifted, it.axis, SymInt.mul(SymInt.sub(it.extent, 1), it.stride));
      kept.push(iter(it.extent, -stride, it.axis));
      continue;
    }
    kept.push(it);
  }
  return { replica: absorbMultiples(kept, prover), offset: shifted };
}

function constOfStride(x: SymExpr): number | null {
  return typeof x === 'number' ? x : null;
}

function absorbedExtent(lo: Iter, hi: Iter, prover: LayoutProver | null): SymExpr | null {
  if (proveEqual(hi.stride, SymInt.mul(lo.extent, lo.stride), prover)) return SymInt.mul(lo.extent, hi.extent);
  const strideLo = constOfStride(lo.stride);
  const strideHi = constOfStride(hi.stride);
  const extentLo = constOfStride(lo.extent);
  const extentHi = constOfStride(hi.extent);
  if (strideLo === null || strideHi === null || extentLo === null || extentHi === null) return null;
  if (strideLo <= 0 || strideHi % strideLo !== 0) return null;
  const q = strideHi / strideLo;
  if (q < 1 || q > extentLo) return null;
  return extentLo + q * (extentHi - 1);
}

function absorbMultiples(replica: readonly Iter[], prover: LayoutProver | null): Iter[] {
  const out = [...replica];
  for (let merged = true; merged; ) {
    merged = false;
    for (let i = 0; i < out.length && !merged; i++) {
      for (let j = 0; j < out.length && !merged; j++) {
        if (i === j || out[i].axis !== out[j].axis) continue;
        const extent = absorbedExtent(out[i], out[j], prover);
        if (extent === null) continue;
        out[i] = iter(extent, out[i].stride, out[i].axis);
        out.splice(j, 1);
        merged = true;
      }
    }
  }
  return out.sort(compareIters);
}

function compareIters(a: Iter, b: Iter): number {
  if (a.axis !== b.axis) return a.axis < b.axis ? -1 : 1;
  const sa = constOfStride(a.stride);
  const sb = constOfStride(b.stride);
  if (sa !== null && sb !== null && sa !== sb) return sa - sb;
  return compareText(`${a.stride}`, `${b.stride}`) || compareText(`${a.extent}`, `${b.extent}`);
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

export class AxeLayout {
  readonly shard: readonly Iter[];
  readonly replica: readonly Iter[];
  readonly offset: Coord;
  private _hash: number | null;

  constructor(shard: readonly Iter[], replica: readonly Iter[] = [], offset: Coord = EMPTY_COORD) {
    for (const it of shard.concat(replica)) {
      if (it.stride === 0) throw new Error('AxeLayout: an iter stride must be non-zero; use a replica iter for a broadcast');
    }
    this.shard = Object.freeze([...shard]);
    this.replica = Object.freeze([...replica]);
    this.offset = offset;
    this._hash = null;
  }

  static rowMajor(shape: readonly SymExpr[], axis: AxeAxisName = AxeAxis.MEM): AxeLayout {
    const iters: Iter[] = new Array(shape.length);
    let stride: SymExpr = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      iters[i] = iter(shape[i], stride, axis);
      stride = SymInt.mul(stride, shape[i]);
    }
    return new AxeLayout(iters);
  }

  static fromPermutation(order: readonly number[], shape: readonly SymExpr[], axis: AxeAxisName = AxeAxis.MEM): AxeLayout {
    const strides: SymExpr[] = new Array(order.length);
    let stride: SymExpr = 1;
    for (let i = order.length - 1; i >= 0; i--) {
      strides[order[i]] = stride;
      stride = SymInt.mul(stride, shape[order[i]]);
    }
    return new AxeLayout(shape.map((extent, dim) => iter(extent, strides[dim], axis)));
  }

  get rank(): number {
    return this.shard.length;
  }

  get domainExtent(): SymExpr {
    return product(this.shard);
  }

  get replicaExtent(): SymExpr {
    return product(this.replica);
  }

  axes(): Set<AxeAxisName> {
    const out = new Set<AxeAxisName>();
    for (const it of this.shard) out.add(it.axis);
    for (const it of this.replica) out.add(it.axis);
    for (const axis of this.offset.keys()) out.add(axis);
    return out;
  }

  isStatic(): boolean {
    for (const it of this.shard.concat(this.replica)) {
      if (!isPositiveConst(it.extent) || !isNonZeroConst(it.stride)) return false;
    }
    for (const value of this.offset.values()) {
      if (typeof value !== 'number') return false;
    }
    return true;
  }

  canonicalize(prover: LayoutProver | null = null): AxeLayout {
    const shard = normalizeShard(this.shard, prover);
    const { replica, offset } = normalizeReplica(this.replica, this.offset, prover);
    return new AxeLayout(shard, replica, offset);
  }

  satisfiesGapCondition(): boolean {
    const sorted = [...this.replica].sort(compareIters);
    for (let i = 0; i + 1 < sorted.length; i++) {
      if (sorted[i].axis !== sorted[i + 1].axis) continue;
      const lo = constOfStride(sorted[i].stride);
      const hi = constOfStride(sorted[i + 1].stride);
      const extent = constOfStride(sorted[i].extent);
      if (lo === null || hi === null || extent === null) return false;
      if (hi <= extent * lo) return false;
    }
    return true;
  }

  isIdentity(): boolean {
    const c = this.canonicalize();
    if (c.replica.length > 0 || c.offset.size > 0) return false;
    let expected: SymExpr = 1;
    for (let i = c.shard.length - 1; i >= 0; i--) {
      if (c.shard[i].axis !== AxeAxis.MEM) return false;
      if (!SymInt.equals(c.shard[i].stride, expected)) return false;
      expected = SymInt.mul(expected, c.shard[i].extent);
    }
    return true;
  }

  applyIters(indices: readonly SymExpr[]): Coord[] {
    if (indices.length !== this.shard.length) {
      throw new Error(`AxeLayout.applyIters expects ${this.shard.length} indices, got ${indices.length}`);
    }
    const base = new Map(this.offset);
    for (let i = 0; i < this.shard.length; i++) {
      coordAddTerm(base, this.shard[i].axis, SymInt.mul(indices[i], this.shard[i].stride));
    }
    return this._replicate(base);
  }

  applyFlat(index: number): Coord[] {
    const indices: number[] = new Array(this.shard.length);
    let rest = index;
    for (let i = this.shard.length - 1; i >= 0; i--) {
      const extent = this.shard[i].extent;
      if (typeof extent !== 'number') throw new Error('AxeLayout.applyFlat requires constant shard extents');
      indices[i] = rest % extent;
      rest = Math.floor(rest / extent);
    }
    return this.applyIters(indices);
  }

  apply(coords: readonly number[], shape: readonly number[]): Coord[] {
    const blocks = this.group(shape);
    if (!blocks) throw new Error(`AxeLayout.apply: shape [${shape.join(', ')}] does not group this layout`);
    const indices: number[] = [];
    for (let d = 0; d < blocks.length; d++) {
      let rest = coords[d];
      const digits: number[] = new Array(blocks[d].length);
      for (let i = blocks[d].length - 1; i >= 0; i--) {
        const extent = blocks[d][i].extent as number;
        digits[i] = rest % extent;
        rest = Math.floor(rest / extent);
      }
      for (const digit of digits) indices.push(digit);
    }
    return new AxeLayout(blocks.flat(), this.replica, this.offset).applyIters(indices);
  }

  private _replicate(base: Coord): Coord[] {
    if (this.replica.length === 0) return [base];
    let out: Coord[] = [base];
    for (const it of this.replica) {
      const extent = it.extent;
      if (typeof extent !== 'number') throw new Error('AxeLayout: replica enumeration requires constant extents');
      const next: Coord[] = [];
      for (const c of out) {
        for (let k = 0; k < extent; k++) {
          const shifted = new Map(c);
          coordAddTerm(shifted, it.axis, SymInt.mul(k, it.stride));
          next.push(shifted);
        }
      }
      out = next;
    }
    return out;
  }

  group(shape: readonly number[]): Iter[][] | null {
    const blocks: Iter[][] = [];
    const pending: Iter[] = [...this.shard];
    let pos = 0;

    for (const dim of shape) {
      const block: Iter[] = [];
      let need = dim;
      while (need !== 1) {
        if (pos >= pending.length) return null;
        const current = pending[pos];
        const extent = typeof current.extent === 'number' ? current.extent : null;
        if (extent === null || extent <= 0) return null;
        if (need % extent === 0) {
          block.push(current);
          need /= extent;
          pos++;
          continue;
        }
        if (extent % need !== 0) return null;
        const [outer, inner] = splitIter(current, need, extent / need);
        block.push(outer);
        pending[pos] = inner;
        need = 1;
      }
      blocks.push(block);
    }
    return pos === pending.length ? blocks : null;
  }

  spanSize(axis: AxeAxisName): number {
    let min = 0;
    let max = 0;
    const offset = this.offset.get(axis) ?? 0;
    if (typeof offset !== 'number') return -1;
    for (const it of this.shard.concat(this.replica)) {
      if (it.axis !== axis) continue;
      if (typeof it.extent !== 'number' || typeof it.stride !== 'number') return -1;
      const edge = (it.extent - 1) * it.stride;
      if (edge < 0) min += edge;
      else max += edge;
    }
    return max - min + 1;
  }

  footprint(): number {
    return this.spanSize(AxeAxis.MEM);
  }

  static tile(a: AxeLayout, shapeA: readonly number[], b: AxeLayout, shapeB: readonly number[]): AxeLayout | null {
    if (shapeA.length !== shapeB.length) return null;
    const blocksA = a.group(shapeA);
    const blocksB = b.group(shapeB);
    if (!blocksA || !blocksB) return null;

    const spans = new Map<AxeAxisName, number>();
    for (const axis of b.axes()) {
      const span = b.spanSize(axis);
      if (span < 0) return null;
      spans.set(axis, span);
    }
    const scale = (it: Iter) => iter(it.extent, SymInt.mul(it.stride, spans.get(it.axis) ?? 1), it.axis);

    const shard: Iter[] = [];
    for (let d = 0; d < blocksA.length; d++) {
      for (const it of blocksA[d]) shard.push(scale(it));
      for (const it of blocksB[d]) shard.push(it);
    }

    const offset = new Map<AxeAxisName, SymExpr>();
    for (const [axis, value] of a.offset) offset.set(axis, SymInt.mul(value, spans.get(axis) ?? 1));
    return new AxeLayout(shard, a.replica.map(scale).concat(b.replica), coordAdd(offset, b.offset));
  }

  slice(region: readonly SliceRegion[], shape: readonly number[]): AxeLayout | null {
    const blocks = this.group(shape);
    if (!blocks || blocks.length !== region.length) return null;

    const shard: Iter[] = [];
    const offset = new Map(this.offset);
    for (let d = 0; d < blocks.length; d++) {
      const block = sliceBlock(blocks[d], region[d], shape[d]);
      if (block === null) return null;
      for (const [axis, value] of block.origin) coordAddTerm(offset, axis, value);
      for (const it of block.iters) shard.push(it);
    }
    return new AxeLayout(shard, this.replica, offset);
  }

  equals(other: unknown, prover: LayoutProver | null = null): boolean {
    if (this === other) return true;
    if (!(other instanceof AxeLayout)) return false;
    const a = this.canonicalize(prover);
    const b = other.canonicalize(prover);

    if (a.shard.length !== b.shard.length) return false;
    for (let i = 0; i < a.shard.length; i++) {
      if (a.shard[i].axis !== b.shard[i].axis) return false;
      if (!proveEqual(a.shard[i].extent, b.shard[i].extent, prover)) return false;
      if (!proveEqual(a.shard[i].stride, b.shard[i].stride, prover)) return false;
    }

    if (a.replica.length === 0 && b.replica.length === 0) return coordEquals(a.offset, b.offset, prover);
    if (a.satisfiesGapCondition() && b.satisfiesGapCondition() && a.replica.length === b.replica.length) {
      let same = coordEquals(a.offset, b.offset, prover);
      for (let i = 0; i < a.replica.length && same; i++) {
        if (a.replica[i].axis !== b.replica[i].axis) same = false;
        else if (!proveEqual(a.replica[i].extent, b.replica[i].extent, prover)) same = false;
        else if (!proveEqual(a.replica[i].stride, b.replica[i].stride, prover)) same = false;
      }
      if (same) return true;
    }
    return differentialEquals(a, b);
  }

  hash(): number {
    if (this._hash !== null) return this._hash;
    const c = this.canonicalize();
    let h = 0x811c9dc5;
    for (const it of c.shard.concat(c.replica)) {
      for (const part of [String(it.extent), String(it.stride), it.axis]) {
        for (let i = 0; i < part.length; i++) h = ((h ^ part.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
      }
    }
    const key = coordKey(c.offset);
    for (let i = 0; i < key.length; i++) h = ((h ^ key.charCodeAt(i)) * 0x01000193) & 0x7fffffff;
    this._hash = h;
    return h;
  }
}

function sliceBlock(
  block: readonly Iter[],
  region: SliceRegion,
  dim: number
): { iters: Iter[]; origin: Coord } | null {
  const { start, extent } = region;
  if (!Number.isInteger(start) || !Number.isInteger(extent)) return null;
  if (start < 0 || extent < 1 || start + extent > dim) return null;

  const extents: number[] = [];
  for (const it of block) {
    if (typeof it.extent !== 'number') return null;
    extents.push(it.extent);
  }

  const digits: number[] = new Array(block.length);
  const origin = new Map<AxeAxisName, SymExpr>();
  let rest = start;
  for (let k = block.length - 1; k >= 0; k--) {
    digits[k] = rest % extents[k];
    rest = Math.floor(rest / extents[k]);
    coordAddTerm(origin, block[k].axis, SymInt.mul(digits[k], block[k].stride));
  }

  const peeled: Iter[] = [];
  let remaining = extent;
  let pivot = block.length - 1;
  for (; pivot >= 0; pivot--) {
    if (digits[pivot] !== 0 || remaining % extents[pivot] !== 0) break;
    peeled.unshift(block[pivot]);
    remaining /= extents[pivot];
  }
  if (remaining === 1) return { iters: peeled, origin };
  if (pivot < 0 || digits[pivot] + remaining > extents[pivot]) return null;
  return { iters: [iter(remaining, block[pivot].stride, block[pivot].axis), ...peeled], origin };
}

function differentialEquals(a: AxeLayout, b: AxeLayout): boolean {
  if (!a.isStatic() || !b.isStatic()) return false;
  const extentA = a.domainExtent;
  const extentB = b.domainExtent;
  if (typeof extentA !== 'number' || extentA !== extentB) return false;
  if (extentA * (a.replicaExtent as number) > DIFFERENTIAL_DOMAIN_LIMIT) return false;
  for (let x = 0; x < extentA; x++) {
    const left = [...new Set(a.applyFlat(x).map(coordKey))].sort();
    const right = [...new Set(b.applyFlat(x).map(coordKey))].sort();
    if (left.length !== right.length) return false;
    for (let i = 0; i < left.length; i++) {
      if (left[i] !== right[i]) return false;
    }
  }
  return true;
}
