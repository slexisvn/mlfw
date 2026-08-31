import { UseDefAnalysis } from './use_def.js';
import { isTerminatorOp } from '../ir/graph/op_traits.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Operation } from '../ir/graph/operation.js';
import type { AnalysisCtor, AnalysisDeps } from './analysis_manager.js';
import type { UseDefResult } from './use_def.js';

type PDomNode = Operation | { opName: string };

export class DominanceResult {
  idom: Map<PDomNode, PDomNode>;
  private _tin: Map<PDomNode, number>;
  private _tout: Map<PDomNode, number>;

  constructor(idom: Map<PDomNode, PDomNode>, topo?: readonly Operation[]) {
    this.idom = idom;
    this._tin = new Map();
    this._tout = new Map();
    this._buildEulerTour();
  }

  _buildEulerTour(): void {
    const children = new Map<PDomNode, PDomNode[]>();
    const roots: PDomNode[] = [];
    for (const [node, parent] of this.idom) {
      let bucket = children.get(parent);
      if (!bucket) { bucket = []; children.set(parent, bucket); }
      bucket.push(node);
    }
    for (const parent of children.keys()) {
      if (!this.idom.has(parent)) roots.push(parent);
    }

    let clock = 0;
    for (const root of roots) {
      const stack: { node: PDomNode; i: number }[] = [{ node: root, i: 0 }];
      this._tin.set(root, clock++);
      while (stack.length > 0) {
        const frame = stack[stack.length - 1];
        const kids = children.get(frame.node);
        if (kids && frame.i < kids.length) {
          const child = kids[frame.i];
          frame.i++;
          if (this._tin.has(child)) continue;
          this._tin.set(child, clock++);
          stack.push({ node: child, i: 0 });
          continue;
        }
        this._tout.set(frame.node, clock++);
        stack.pop();
      }
    }
  }

  postDominates(a: PDomNode, b: PDomNode): boolean {
    if (a === b) return true;
    const aIn = this._tin.get(a);
    const bIn = this._tin.get(b);
    if (aIn === undefined || bIn === undefined) return false;
    return aIn <= bIn && (this._tout.get(b) as number) <= (this._tout.get(a) as number);
  }

  immediatePDom(op: PDomNode): PDomNode | null {
    return this.idom.get(op) || null;
  }

  pathToPDom(from: PDomNode): PDomNode[] {
    const path: PDomNode[] = [];
    let cur: PDomNode | undefined = this.idom.get(from);
    while (cur) {
      path.push(cur);
      if (cur === this.idom.get(cur)) break;
      cur = this.idom.get(cur);
    }
    return path;
  }
}

export class PostDominanceAnalysis {
  static get name(): string { return 'post_dominance'; }
  static get depKey(): string { return 'postDominance'; }
  static get dependencies(): readonly AnalysisCtor[] { return [UseDefAnalysis as unknown as AnalysisCtor]; }

  static compute(func: GraphFunction, deps: AnalysisDeps = {}): DominanceResult {
    const useDef = (deps.useDef as UseDefResult | undefined) || UseDefAnalysis.compute(func);
    const topo = useDef.topologicalOrder;

    const succs = new Map<Operation, Operation[]>();
    for (const op of topo) {
      succs.set(op, []);
    }

    for (const op of topo) {
      for (let i = 0; i < op.numResults; i++) {
        const val = op.getResult(i);
        for (const use of val.uses()) {
          const user = use.user;
          if (succs.has(user)) {
            (succs.get(op) as Operation[]).push(user);
          }
        }
      }
    }

    const exitSet = new Set<Operation>();
    for (const op of topo) {
      if (isTerminatorOp(op.opName)) {
        exitSet.add(op);
      }
    }

    const sentinel: PDomNode = { opName: '__pdom_root__' };
    const levels = Math.max(1, Math.ceil(Math.log2(topo.length + 2)) + 1);
    const idom = new Map<PDomNode, PDomNode>();
    const depth = new Map<PDomNode, number>();
    const up = new Map<PDomNode, PDomNode[]>();

    depth.set(sentinel, 0);
    up.set(sentinel, new Array(levels).fill(sentinel));

    const link = (node: PDomNode, parent: PDomNode): void => {
      idom.set(node, parent);
      depth.set(node, (depth.get(parent) as number) + 1);
      const table = new Array<PDomNode>(levels);
      table[0] = parent;
      for (let k = 1; k < levels; k++) table[k] = (up.get(table[k - 1]) as PDomNode[])[k - 1];
      up.set(node, table);
    };

    const lca = (u: PDomNode, v: PDomNode): PDomNode => {
      if ((depth.get(u) as number) < (depth.get(v) as number)) { const t = u; u = v; v = t; }
      let diff = (depth.get(u) as number) - (depth.get(v) as number);
      for (let k = 0; k < levels; k++) {
        if ((diff >> k) & 1) u = (up.get(u) as PDomNode[])[k];
      }
      if (u === v) return u;
      for (let k = levels - 1; k >= 0; k--) {
        if ((up.get(u) as PDomNode[])[k] !== (up.get(v) as PDomNode[])[k]) {
          u = (up.get(u) as PDomNode[])[k];
          v = (up.get(v) as PDomNode[])[k];
        }
      }
      return (up.get(u) as PDomNode[])[0];
    };

    for (let i = topo.length - 1; i >= 0; i--) {
      const op = topo[i];
      if (exitSet.has(op)) {
        link(op, sentinel);
        continue;
      }
      let parent: PDomNode | null = null;
      for (const s of succs.get(op) as Operation[]) {
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
