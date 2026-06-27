export function topoSortByOperands(ops, contains, onCycle = 'ignore') {
  const ordered = [];
  const state = new Map();
  for (const root of ops) {
    if (state.get(root) !== undefined) continue;
    state.set(root, 1);
    const stack = [{ op: root, i: 0 }];
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const op = frame.op;
      if (frame.i < op.numOperands) {
        const def = op.getOperand(frame.i).definingOp;
        frame.i++;
        if (def && contains(def)) {
          const ds = state.get(def);
          if (ds === undefined) {
            state.set(def, 1);
            stack.push({ op: def, i: 0 });
          } else if (ds === 1) {
            if (onCycle === 'throw') throw new Error('topo sort: cycle detected');
            if (onCycle === 'null') return null;
          }
        }
        continue;
      }
      state.set(op, 2);
      ordered.push(op);
      stack.pop();
    }
  }
  return ordered;
}

export function topoSortOpSet(ops, onCycle = 'throw') {
  const arr = Array.isArray(ops) ? ops : [...ops];
  const set = new Set(arr);
  return topoSortByOperands(arr, (op) => set.has(op), onCycle);
}
