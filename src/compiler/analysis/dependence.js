import { AccessKind } from './buffer_access.js';
import { mixedRadixDecomposition } from './iter_map.js';

export const Direction = Object.freeze({ LT: 1, EQ: 2, GT: 4 });
export const ANY_DIRECTION = Direction.LT | Direction.EQ | Direction.GT;

export const DepKind = Object.freeze({
  RAW: 'RAW',
  WAR: 'WAR',
  WAW: 'WAW',
  OPAQUE: 'OPAQUE',
});

export class Dependence {
  constructor(buffer, src, dst, kind, loops, masks) {
    this.buffer = buffer;
    this.src = src;
    this.dst = dst;
    this.kind = kind;
    this.loops = loops;
    this.masks = masks;
  }

  get isLoopIndependent() {
    return this.masks.every((m) => m === Direction.EQ);
  }
}

function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

function commonNest(srcSpace, dstSpace) {
  const n = Math.min(srcSpace.length, dstSpace.length);
  const loops = [];
  for (let i = 0; i < n; i++) {
    const a = srcSpace[i];
    const b = dstSpace[i];
    if (a === null || b === null || a.node !== b.node) break;
    loops.push(a);
  }
  return loops;
}

function coefficients(form, levelIndex) {
  const coeffs = new Array(levelIndex.size).fill(0);
  let foreign = false;
  for (const [name, coeff] of form.terms) {
    const level = levelIndex.get(name);
    if (level === undefined) foreign = true;
    else coeffs[level] = coeff;
  }
  return { coeffs, foreign };
}

const INDEPENDENT = Symbol('independent');

function subscriptDirections(srcForm, dstForm, loops, levelIndex, varRanges) {
  const n = loops.length;
  if (!srcForm || !dstForm) return new Array(n).fill(ANY_DIRECTION);

  const src = coefficients(srcForm, levelIndex);
  const dst = coefficients(dstForm, levelIndex);
  const delta = dstForm.offset - srcForm.offset;

  const involved = [];
  for (let k = 0; k < n; k++) {
    if (src.coeffs[k] !== 0 || dst.coeffs[k] !== 0) involved.push(k);
  }

  if (src.foreign || dst.foreign) return new Array(n).fill(ANY_DIRECTION);

  if (involved.length === 0) return delta === 0 ? new Array(n).fill(ANY_DIRECTION) : INDEPENDENT;

  const masks = new Array(n).fill(ANY_DIRECTION);

  if (involved.length === 1) {
    const k = involved[0];
    const a = src.coeffs[k];
    const b = dst.coeffs[k];
    const extent = loops[k].extent;

    if (a === b) {
      if (delta % a !== 0) return INDEPENDENT;
      const distance = -delta / a;
      if (Math.abs(distance) >= extent) return INDEPENDENT;
      masks[k] = distance > 0 ? Direction.LT : (distance === 0 ? Direction.EQ : Direction.GT);
      return masks;
    }

    if (b === 0) {
      if (delta % a !== 0) return INDEPENDENT;
      const iv = delta / a;
      if (iv < loops[k].min || iv >= loops[k].min + extent) return INDEPENDENT;
      return masks;
    }

    if (a === 0) {
      if (delta % b !== 0) return INDEPENDENT;
      const jv = -delta / b;
      if (jv < loops[k].min || jv >= loops[k].min + extent) return INDEPENDENT;
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

function dependenceKind(src, dst) {
  if (src.kind === AccessKind.WRITE) {
    return dst.kind === AccessKind.WRITE ? DepKind.WAW : DepKind.RAW;
  }
  return DepKind.WAR;
}

export function accessDependence(src, dst) {
  const loops = commonNest(src.iterSpace, dst.iterSpace);
  const levelIndex = new Map();
  const varRanges = new Map();
  for (let i = 0; i < loops.length; i++) {
    levelIndex.set(loops[i].name, i);
    varRanges.set(loops[i].name, [loops[i].min, loops[i].extent]);
  }

  const rank = Math.max(src.forms.length, dst.forms.length);
  const masks = new Array(loops.length).fill(ANY_DIRECTION);

  for (let d = 0; d < rank; d++) {
    const dims = subscriptDirections(src.forms[d], dst.forms[d], loops, levelIndex, varRanges);
    if (dims === INDEPENDENT) return null;
    for (let k = 0; k < loops.length; k++) {
      masks[k] &= dims[k];
      if (masks[k] === 0) return null;
    }
  }

  return new Dependence(src.buffer, src, dst, dependenceKind(src, dst), loops, masks);
}

export function bufferDependences(accesses) {
  const deps = [];
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

export function dependences(byBuffer) {
  const deps = [];
  for (const [, accesses] of byBuffer) {
    for (const dep of bufferDependences(accesses)) deps.push(dep);
  }
  return deps;
}

export function carriesDependence(deps, loopNode) {
  for (const dep of deps) {
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

function windowViolation(masks, afterOrder) {
  const n = masks.length;
  const free = new Set();
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

export function permutationPreservesDependences(deps, before, after) {
  const windowIndex = new Map();
  for (let i = 0; i < before.length; i++) windowIndex.set(before[i], i);
  const afterOrder = after.map((l) => windowIndex.get(l));

  for (const dep of deps) {
    const masks = new Array(before.length).fill(0);
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
