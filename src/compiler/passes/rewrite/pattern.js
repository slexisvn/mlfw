import { PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { TraceLevel } from '../../pipeline/trace.js';

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

  applyPatterns(func, maxIterations = 10, trace = null) {
    const builder = new IRBuilder(func);
    let totalRewrites = 0;

    const worklist = [...func.opsRecursive()];
    let head = 0;
    const queued = new Set(worklist);
    const enqueue = (op) => {
      if (!op || !op.parentBlock || queued.has(op)) return;
      queued.add(op);
      worklist.push(op);
    };

    const safetyBudget = Math.max(maxIterations, 1) * Math.max(worklist.length, 1) * 4 + 1000;
    let steps = 0;
    let capped = false;

    while (head < worklist.length) {
      if (++steps > safetyBudget) { capped = true; break; }
      const op = worklist[head++];
      queued.delete(op);
      if (!op.parentBlock) continue;

      const patterns = this.patternSet.getForOp(op.opName);
      for (const pattern of patterns) {
        if (!pattern.match(op)) continue;

        const block = op.parentBlock;
        const prevOp = op._prev;
        const nextOp = op._next;
        const affected = [];
        for (let r = 0; r < op.numResults; r++) {
          for (const u of op.getResult(r).getUsers()) affected.push(u);
        }
        for (let o = 0; o < op.numOperands; o++) {
          const d = op.getOperand(o).definingOp;
          if (d) affected.push(d);
        }

        builder.block = block;
        builder.setInsertionPoint(op);
        if (!pattern.rewrite(op, builder)) continue;

        totalRewrites++;
        for (const a of affected) enqueue(a);
        let cur = prevOp ? prevOp._next : block._head;
        let guard = block._size + 2;
        while (cur && cur !== nextOp && guard-- > 0) { enqueue(cur); cur = cur._next; }
        enqueue(op);
        break;
      }
    }

    if (trace) {
      if (capped && trace.level >= TraceLevel.INFO) {
        trace.emit({
          type: 'pass_detail', passName: 'PatternApplicator',
          message: `pattern rewriting hit safety budget (${safetyBudget}) without converging`,
          totalRewrites, level: TraceLevel.INFO,
        });
      }
      if (trace.level >= TraceLevel.DEBUG && totalRewrites > 0) {
        trace.emit({
          type: 'pass_detail', passName: 'PatternApplicator',
          totalRewrites, patternCount: this.patternSet.patterns.length,
          level: TraceLevel.DEBUG,
        });
      }
    }

    return totalRewrites > 0 ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
