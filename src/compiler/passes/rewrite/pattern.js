import { PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';

export class Pattern {
  constructor(name, benefit = 1) {
    this.name = name;
    this.benefit = benefit;
    this.rootOpName = null;
  }

  match(op) { return false; }
  rewrite(op, builder) { return false; }
}

export class PatternSet {
  constructor() {
    this.patterns = [];
    this._byOp = new Map();
    this._generic = [];
    this._sorted = false;
  }

  add(pattern) {
    this.patterns.push(pattern);
    this._sorted = false;
    if (pattern.rootOpName) {
      let list = this._byOp.get(pattern.rootOpName);
      if (!list) {
        list = [];
        this._byOp.set(pattern.rootOpName, list);
      }
      list.push(pattern);
    } else {
      this._generic.push(pattern);
    }
  }

  _ensureSorted() {
    if (this._sorted) return;
    const cmp = (a, b) => b.benefit - a.benefit;
    for (const [, list] of this._byOp) {
      list.sort(cmp);
    }
    this._generic.sort(cmp);
    this._sorted = true;
  }

  get() {
    return [...this.patterns].sort((a, b) => b.benefit - a.benefit);
  }

  getForOp(opName) {
    this._ensureSorted();
    const specific = this._byOp.get(opName);
    if (!specific) return this._generic;
    if (this._generic.length === 0) return specific;
    const merged = new Array(specific.length + this._generic.length);
    let si = 0, gi = 0, mi = 0;
    while (si < specific.length && gi < this._generic.length) {
      if (specific[si].benefit >= this._generic[gi].benefit) {
        merged[mi++] = specific[si++];
      } else {
        merged[mi++] = this._generic[gi++];
      }
    }
    while (si < specific.length) merged[mi++] = specific[si++];
    while (gi < this._generic.length) merged[mi++] = this._generic[gi++];
    return merged;
  }

  hasPatterns() {
    return this.patterns.length > 0;
  }
}

export class PatternApplicator {
  constructor(patternSet) {
    this.patternSet = patternSet;
  }

  applyPatterns(func, maxIterations = 10) {
    let changed = false;
    let iteration = 0;
    const builder = new IRBuilder(func);

    while (iteration < maxIterations) {
      let iterChanged = false;
      for (const op of [...func.opsArray()]) {
        if (!op.parentBlock) continue;

        const patterns = this.patternSet.getForOp(op.opName);
        for (const pattern of patterns) {
          if (pattern.match(op)) {
            builder.block = op.parentBlock;
            builder.setInsertionPoint(op);
            if (pattern.rewrite(op, builder)) {
              iterChanged = true;
              changed = true;
              break;
            }
          }
        }
      }
      if (!iterChanged) break;
      iteration++;
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
