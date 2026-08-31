import { effectPredecessors } from '../ir/graph/op_traits.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { AnalysisCtor } from './analysis_manager.js';

export class UseDefResult {
  topologicalOrder: Operation[];
  valueToOp: Map<Value, Operation>;
  opUsers: Map<Operation, Set<Operation>>;
  depth: Map<Operation, number>;
  height: Map<Operation, number>;

  constructor(topologicalOrder: Operation[], valueToOp: Map<Value, Operation>, opUsers: Map<Operation, Set<Operation>>, depth: Map<Operation, number>, height: Map<Operation, number>) {
    this.topologicalOrder = topologicalOrder;
    this.valueToOp = valueToOp;
    this.opUsers = opUsers;
    this.depth = depth;
    this.height = height;
  }
}

export class UseDefAnalysis {
  static get name(): string { return 'use_def'; }
  static get depKey(): string { return 'useDef'; }
  static get dependencies(): readonly AnalysisCtor[] { return []; }

  static compute(func: GraphFunction): UseDefResult {
    const topologicalOrder: Operation[] = [];
    const valueToOp = new Map<Value, Operation>();
    const opUsers = new Map<Operation, Set<Operation>>();

    const visitedOps = new Set<Operation>();
    const visitingOps = new Set<Operation>();
    const effectPred = effectPredecessors(func.ops());

    const visit = (root: Operation): void => {
      if (visitedOps.has(root)) return;
      visitingOps.add(root);
      const stack: { op: Operation; i: number }[] = [{ op: root, i: 0 }];
      while (stack.length > 0) {
        const frame = stack[stack.length - 1];
        const op = frame.op;
        const numDeps = op.numOperands + (effectPred.has(op) ? 1 : 0);
        if (frame.i < numDeps) {
          const defOp = frame.i < op.numOperands
            ? op.getOperand(frame.i).definingOp
            : effectPred.get(op) as Operation;
          frame.i++;
          if (defOp && !visitedOps.has(defOp)) {
            if (visitingOps.has(defOp)) {
              throw new Error('Cycle detected in UseDefAnalysis');
            }
            visitingOps.add(defOp);
            stack.push({ op: defOp, i: 0 });
          }
          continue;
        }
        visitingOps.delete(op);
        visitedOps.add(op);
        topologicalOrder.push(op);
        stack.pop();
      }
    };

    for (const op of func.ops()) {
      for (let i = 0; i < op.numResults; i++) {
        valueToOp.set(op.getResult(i), op);
      }
      opUsers.set(op, new Set());
    }

    const depth = new Map<Operation, number>();
    const height = new Map<Operation, number>();

    for (const op of func.ops()) {
      visit(op);
    }

    for (const op of topologicalOrder) {
      let maxDepth = 0;
      for (let i = 0; i < op.numOperands; i++) {
        const defOp = op.getOperand(i).definingOp;
        if (defOp) {
          (opUsers.get(defOp) as Set<Operation>).add(op);
          const d = depth.get(defOp) || 0;
          if (d + 1 > maxDepth) maxDepth = d + 1;
        }
      }
      depth.set(op, maxDepth);
    }

    for (let i = topologicalOrder.length - 1; i >= 0; i--) {
      const op = topologicalOrder[i];
      let maxHeight = 0;
      const users = opUsers.get(op) || new Set();
      for (const user of users) {
        const h = height.get(user) || 0;
        if (h + 1 > maxHeight) maxHeight = h + 1;
      }
      height.set(op, maxHeight);
    }

    return new UseDefResult(topologicalOrder, valueToOp, opUsers, depth, height);
  }
}
