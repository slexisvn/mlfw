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

    const exitSet = new Set();
    for (const op of topo) {
      if (op.opName === 'return' || op.opName === 'yield') {
        exitSet.add(op);
      }
    }

    const sentinel = { opName: '__pdom_root__' };
    const levels = Math.max(1, Math.ceil(Math.log2(topo.length + 2)) + 1);
    const idom = new Map();
    const depth = new Map();
    const up = new Map();

    depth.set(sentinel, 0);
    up.set(sentinel, new Array(levels).fill(sentinel));

    const link = (node, parent) => {
      idom.set(node, parent);
      depth.set(node, depth.get(parent) + 1);
      const table = new Array(levels);
      table[0] = parent;
      for (let k = 1; k < levels; k++) table[k] = up.get(table[k - 1])[k - 1];
      up.set(node, table);
    };

    const lca = (u, v) => {
      if (depth.get(u) < depth.get(v)) { const t = u; u = v; v = t; }
      let diff = depth.get(u) - depth.get(v);
      for (let k = 0; k < levels; k++) {
        if ((diff >> k) & 1) u = up.get(u)[k];
      }
      if (u === v) return u;
      for (let k = levels - 1; k >= 0; k--) {
        if (up.get(u)[k] !== up.get(v)[k]) {
          u = up.get(u)[k];
          v = up.get(v)[k];
        }
      }
      return up.get(u)[0];
    };

    for (let i = topo.length - 1; i >= 0; i--) {
      const op = topo[i];
      if (exitSet.has(op)) {
        link(op, sentinel);
        continue;
      }
      let parent = null;
      for (const s of succs.get(op)) {
        if (!idom.has(s)) continue;
        parent = parent === null ? s : lca(parent, s);
      }
      if (parent !== null) link(op, parent);
    }

    for (const [k, v] of idom) {
      if (v === sentinel) idom.delete(k);
    }

    return new DominanceResult(idom, topo);
  }
}
