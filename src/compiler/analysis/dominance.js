import { UseDefAnalysis } from './use_def.js';

export class DominanceResult {
  constructor(idom, topoOrder) {
    this.idom = idom;
    this._topoOrder = topoOrder;
    this._orderIndex = new Map();
    for (let i = 0; i < topoOrder.length; i++) {
      this._orderIndex.set(topoOrder[i], i);
    }
  }

  postDominates(a, b) {
    let cur = b;
    while (cur) {
      if (cur === a) return true;
      cur = this.idom.get(cur);
    }
    return false;
  }

  immediatePDom(op) {
    return this.idom.get(op) || null;
  }

  pathToPDom(from) {
    const path = [];
    let cur = this.idom.get(from);
    while (cur) {
      path.push(cur);
      if (cur === this.idom.get(cur)) break;
      cur = this.idom.get(cur);
    }
    return path;
  }
}

export class PostDominanceAnalysis {
  static get name() { return 'post_dominance'; }
  static get depKey() { return 'postDominance'; }
  static get dependencies() { return [UseDefAnalysis]; }

  static compute(func, deps = {}) {
    const useDef = deps.useDef || UseDefAnalysis.compute(func);
    const topo = useDef.topologicalOrder;

    const succs = new Map();
    for (const op of topo) {
      succs.set(op, []);
    }

    for (const op of topo) {
      for (let i = 0; i < op.numResults; i++) {
        const val = op.getResult(i);
        for (const use of val.uses()) {
          const user = use.user;
          if (succs.has(user)) {
            succs.get(op).push(user);
          }
        }
      }
    }

    const exits = [];
    for (const op of topo) {
      if (op.opName === 'return' || op.opName === 'yield') {
        exits.push(op);
      }
    }

    const sentinel = { opName: '__pdom_root__' };
    const idom = new Map();

    for (const ex of exits) {
      idom.set(ex, sentinel);
    }
    idom.set(sentinel, sentinel);

    const rpo = [];
    for (let i = topo.length - 1; i >= 0; i--) {
      rpo.push(topo[i]);
    }

    const rpoIndex = new Map();
    rpoIndex.set(sentinel, -1);
    for (let i = 0; i < rpo.length; i++) {
      rpoIndex.set(rpo[i], i);
    }

    let changed = true;
    while (changed) {
      changed = false;
      for (const op of rpo) {
        if (idom.get(op) === sentinel && exits.includes(op)) continue;

        const opSuccs = succs.get(op);
        if (!opSuccs || opSuccs.length === 0) continue;

        let newIdom = null;
        for (const s of opSuccs) {
          if (!idom.has(s)) continue;
          if (newIdom === null) {
            newIdom = s;
          } else {
            newIdom = intersect(newIdom, s, idom, rpoIndex);
          }
        }

        if (newIdom === null) continue;
        if (idom.get(op) !== newIdom) {
          idom.set(op, newIdom);
          changed = true;
        }
      }
    }

    idom.delete(sentinel);
    for (const [k, v] of idom) {
      if (v === sentinel) idom.delete(k);
    }

    return new DominanceResult(idom, topo);
  }
}

function intersect(a, b, idom, rpoIndex) {
  let fa = a, fb = b;
  let limit = rpoIndex.size;
  while (fa !== fb && limit-- > 0) {
    const ia = rpoIndex.get(fa) ?? Infinity;
    const ib = rpoIndex.get(fb) ?? Infinity;
    if (ia > ib) {
      fa = idom.get(fa);
      if (!fa) return fb;
    } else {
      fb = idom.get(fb);
      if (!fb) return fa;
    }
  }
  return fa;
}
