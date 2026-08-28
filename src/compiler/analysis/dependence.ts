import { AccessKind, isStaticLevel } from './buffer_access.js';
import { mixedRadixDecomposition } from './iter_map.js';
import { gcd } from '../../util/integer.js';
import type { LinearForm, VarRange } from './iter_map.js';
import type { BufferAccess, IterLevel } from './buffer_access.js';
import type { Buffer } from '../ir/tensor/buffer.js';
import type { TirNode } from '../ir/tensor/nodes.js';

export const Direction = Object.freeze({ LT: 1, EQ: 2, GT: 4 });
export const ANY_DIRECTION = Direction.LT | Direction.EQ | Direction.GT;

export const DepKind = Object.freeze({
  RAW: 'RAW',
  WAR: 'WAR',
  WAW: 'WAW',
  OPAQUE: 'OPAQUE',
});

export type DirectionMask = number;
export type DepKindValue = (typeof DepKind)[keyof typeof DepKind];
type Coefficients = { coeffs: number[]; foreign: boolean };

export class Dependence {
  buffer: Buffer;
  src: BufferAccess;
  dst: BufferAccess;
  kind: DepKindValue;
  loops: IterLevel[];
  masks: DirectionMask[];

  constructor(buffer: Buffer, src: BufferAccess, dst: BufferAccess, kind: DepKindValue, loops: IterLevel[], masks: DirectionMask[]) {
    this.buffer = buffer;
    this.src = src;
    this.dst = dst;
    this.kind = kind;
    this.loops = loops;
    this.masks = masks;
  }

  get isLoopIndependent(): boolean {
    return this.masks.every((m) => m === Direction.EQ);
  }
}

function commonNest(srcSpace: readonly IterLevel[], dstSpace: readonly IterLevel[]): IterLevel[] {
  const n = Math.min(srcSpace.length, dstSpace.length);
  const loops: IterLevel[] = [];
  for (let i = 0; i < n; i++) {
    if (srcSpace[i].node !== dstSpace[i].node) break;
    loops.push(srcSpace[i]);
  }
  return loops;
}

function coefficients(form: LinearForm, levelIndex: ReadonlyMap<string, number>, levels: number): Coefficients {
  const coeffs: number[] = new Array(levels).fill(0);
  let foreign = false;
  for (const [name, coeff] of form.terms) {
    const level = levelIndex.get(name);
    if (level === undefined) foreign = true;
    else coeffs[level] = coeff;
  }
  return { coeffs, foreign };
}

const INDEPENDENT = Symbol('independent');

function subscriptDirections(srcForm: LinearForm | null | undefined, dstForm: LinearForm | null | undefined, loops: readonly IterLevel[], levelIndex: ReadonlyMap<string, number>, varRanges: ReadonlyMap<string, VarRange>): DirectionMask[] | typeof INDEPENDENT {
  const n = loops.length;
  if (!srcForm || !dstForm) return new Array<DirectionMask>(n).fill(ANY_DIRECTION);

  const src = coefficients(srcForm, levelIndex, n);
  const dst = coefficients(dstForm, levelIndex, n);
  const delta = dstForm.offset - srcForm.offset;

  const involved: number[] = [];
  for (let k = 0; k < n; k++) {
    if (src.coeffs[k] !== 0 || dst.coeffs[k] !== 0) involved.push(k);
  }

  if (src.foreign || dst.foreign) return new Array<DirectionMask>(n).fill(ANY_DIRECTION);

  if (involved.length === 0) return delta === 0 ? new Array<DirectionMask>(n).fill(ANY_DIRECTION) : INDEPENDENT;

  const masks: DirectionMask[] = new Array<DirectionMask>(n).fill(ANY_DIRECTION);

  if (involved.length === 1) {
    const k = involved[0];
    const a = src.coeffs[k];
    const b = dst.coeffs[k];
    const { min, extent } = loops[k];
    const ranged = min !== null && extent !== null;

    if (a === b) {
      if (delta % a !== 0) return INDEPENDENT;
      const distance = -delta / a;
      if (ranged && Math.abs(distance) >= extent) return INDEPENDENT;
      masks[k] = distance > 0 ? Direction.LT : (distance === 0 ? Direction.EQ : Direction.GT);
      return masks;
    }

    if (b === 0) {
      if (delta % a !== 0) return INDEPENDENT;
      const iv = delta / a;
      if (ranged && (iv < min || iv >= min + extent)) return INDEPENDENT;
      return masks;
    }

    if (a === 0) {
      if (delta % b !== 0) return INDEPENDENT;
      const jv = -delta / b;
      if (ranged && (jv < min || jv >= min + extent)) return INDEPENDENT;
      return masks;
    }

    if (delta % gcd(a, b) !== 0) return INDEPENDENT;
    return masks;
  }

  const uniform = involved.every((k) => src.coeffs[k] === dst.coeffs[k]);
  if (uniform && delta === 0 && mixedRadixDecomposition(srcForm, varRanges) !== null) {
    for (const k of involved) masks[k] = Direction.EQ;
    return masks;
  }

  let g = 0;
  for (const k of involved) g = gcd(g, gcd(src.coeffs[k], dst.coeffs[k]));
  if (g !== 0 && delta % g !== 0) return INDEPENDENT;
  return masks;
}

function dependenceKind(src: BufferAccess, dst: BufferAccess): DepKindValue {
  if (src.kind === AccessKind.WRITE) {
    return dst.kind === AccessKind.WRITE ? DepKind.WAW : DepKind.RAW;
  }
  return DepKind.WAR;
}

function negateMask(mask: DirectionMask): DirectionMask {
  let out = mask & Direction.EQ;
  if (mask & Direction.LT) out |= Direction.GT;
  if (mask & Direction.GT) out |= Direction.LT;
  return out;
}

function isLexNegative(masks: readonly DirectionMask[]): boolean {
  for (const mask of masks) {
    if (mask === Direction.EQ) continue;
    return mask === Direction.GT;
  }
  return false;
}

export function accessDependence(src: BufferAccess, dst: BufferAccess): Dependence | null {
  const loops = commonNest(src.iterSpace, dst.iterSpace);
  const levelIndex = new Map<string, number>();
  const varRanges = new Map<string, VarRange>();
  for (let i = 0; i < loops.length; i++) {
    const level = loops[i];
    if (level.name === null) continue;
    levelIndex.set(level.name, i);
    if (isStaticLevel(level)) varRanges.set(level.name, [level.min, level.extent]);
  }

  const rank = Math.max(src.forms.length, dst.forms.length);
  const masks: DirectionMask[] = new Array<DirectionMask>(loops.length).fill(ANY_DIRECTION);

  for (let d = 0; d < rank; d++) {
    const dims = subscriptDirections(src.forms[d], dst.forms[d], loops, levelIndex, varRanges);
    if (dims === INDEPENDENT) return null;
    for (let k = 0; k < loops.length; k++) {
      masks[k] &= dims[k];
      if (masks[k] === 0) return null;
    }
  }

  if (isLexNegative(masks)) {
    return new Dependence(src.buffer, dst, src, dependenceKind(dst, src), loops, masks.map(negateMask));
  }

  return new Dependence(src.buffer, src, dst, dependenceKind(src, dst), loops, masks);
}

export function bufferDependences(accesses: readonly BufferAccess[]): Dependence[] {
  const deps: Dependence[] = [];
  for (const write of accesses) {
    if (write.kind !== AccessKind.WRITE) continue;
    for (const other of accesses) {
      if (other.kind === AccessKind.WRITE && other.position < write.position) continue;
      const src = write.position <= other.position ? write : other;
      const dst = write.position <= other.position ? other : write;
      const dep = accessDependence(src, dst);
      if (dep) deps.push(dep);
    }
  }
  return deps;
}

export function dependences(byBuffer: ReadonlyMap<Buffer, readonly BufferAccess[]>): Dependence[] {
  const deps: Dependence[] = [];
  for (const [, accesses] of byBuffer) {
    for (const dep of bufferDependences(accesses)) deps.push(dep);
  }
  return deps;
}

export function carriesDependence(deps: readonly Dependence[], loopNode: TirNode, ignore?: (buffer: Buffer) => boolean): Dependence | null {
  for (const dep of deps) {
    if (ignore !== undefined && ignore(dep.buffer)) continue;
    const level = dep.loops.findIndex((l) => l.node === loopNode);
    if (level < 0) continue;
    let outerEq = true;
    for (let m = 0; m < level; m++) {
      if (!(dep.masks[m] & Direction.EQ)) { outerEq = false; break; }
    }
    if (!outerEq) continue;
    if (dep.masks[level] & (Direction.LT | Direction.GT)) return dep;
  }
  return null;
}

function windowViolation(masks: readonly DirectionMask[], afterOrder: readonly number[]): boolean {
  const n = masks.length;
  const free = new Set<number>();
  for (let p = n - 1; p >= 0; p--) {
    const level = afterOrder[p];
    if (masks[level] & Direction.GT) {
      let prefixEq = true;
      for (let m = 0; m < p; m++) {
        if (!(masks[afterOrder[m]] & Direction.EQ)) { prefixEq = false; break; }
      }
      if (prefixEq) {
        for (let b = 0; b < level; b++) {
          if (!free.has(b)) continue;
          if (masks[b] & Direction.LT) return true;
          if (!(masks[b] & Direction.EQ)) break;
        }
      }
    }
    free.add(level);
  }
  return false;
}

export function permutationPreservesDependences(deps: readonly Dependence[], before: readonly TirNode[], after: readonly TirNode[]): Dependence | null {
  const windowIndex = new Map<TirNode, number>();
  for (let i = 0; i < before.length; i++) windowIndex.set(before[i], i);
  const afterOrder = after.map((l) => windowIndex.get(l) as number);

  for (const dep of deps) {
    const masks: DirectionMask[] = new Array<DirectionMask>(before.length).fill(0);
    let covered = 0;
    let outerEq = true;
    for (let i = 0; i < dep.loops.length; i++) {
      const w = windowIndex.get(dep.loops[i].node);
      if (w === undefined) {
        if (covered === 0) {
          if (!(dep.masks[i] & Direction.EQ)) { outerEq = false; break; }
        }
        continue;
      }
      masks[w] = dep.masks[i];
      covered++;
    }
    if (!outerEq || covered === 0) continue;
    if (covered !== before.length) return dep;
    if (windowViolation(masks, afterOrder)) return dep;
  }
  return null;
}
