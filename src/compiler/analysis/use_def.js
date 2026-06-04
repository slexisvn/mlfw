export class UseDefResult {
  constructor(topologicalOrder, valueToOp, opUsers, depth, height) {
    this.topologicalOrder = topologicalOrder;
    this.valueToOp = valueToOp;
    this.opUsers = opUsers;
    this.depth = depth;
    this.height = height;
  }
}

export class UseDefAnalysis {
  static get name() { return 'use_def'; }
  static get depKey() { return 'useDef'; }
  static get dependencies() { return []; }

  static compute(func) {
    const topologicalOrder = [];
    const valueToOp = new Map();
    const opUsers = new Map();

    const visitedOps = new Set();
    const visitingOps = new Set();

    const visit = (op) => {
      if (visitedOps.has(op)) return;
      if (visitingOps.has(op)) {
        throw new Error('Cycle detected in UseDefAnalysis');
      }
      visitingOps.add(op);

      for (let i = 0; i < op.numOperands; i++) {
        const defOp = op.getOperand(i).definingOp;
        if (defOp) visit(defOp);
      }

      visitingOps.delete(op);
      visitedOps.add(op);
      topologicalOrder.push(op);
    };

    for (const op of func.ops()) {
      for (let i = 0; i < op.numResults; i++) {
        valueToOp.set(op.getResult(i), op);
      }
      opUsers.set(op, new Set());
    }

    const depth = new Map();
    const height = new Map();

    for (const op of func.ops()) {
      visit(op);
    }

    for (const op of topologicalOrder) {
      let maxDepth = 0;
      for (let i = 0; i < op.numOperands; i++) {
        const defOp = op.getOperand(i).definingOp;
        if (defOp) {
          opUsers.get(defOp).add(op);
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
