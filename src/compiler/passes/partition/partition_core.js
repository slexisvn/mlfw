export function topoSortOps(ops) {
  const opSet = new Set(ops);
  const ordered = [];
  const state = new Map();
  const visit = (op) => {
    const s = state.get(op);
    if (s === 2) return;
    if (s === 1) throw new Error('partition topo sort: cycle detected');
    state.set(op, 1);
    for (let i = 0; i < op.numOperands; i++) {
      const d = op.getOperand(i).definingOp;
      if (d && opSet.has(d)) visit(d);
    }
    state.set(op, 2);
    ordered.push(op);
  };
  for (const op of ops) visit(op);
  return ordered;
}

export function buildPartitions(partitionOps, { labelOf, sameLabel = (a, b) => a === b, canMerge = () => true, onAttach = () => {}, sort = topoSortOps }) {
  const topo = sort(partitionOps);
  const opToPart = new Map();
  const preds = new Map();
  const partitions = [];
  let nextId = 0;

  const isUpstreamOf = (ancestor, node) => {
    if (ancestor === node) return true;
    const stack = [node];
    const seen = new Set();
    while (stack.length > 0) {
      const cur = stack.pop();
      if (cur === ancestor) return true;
      if (seen.has(cur)) continue;
      seen.add(cur);
      const p = preds.get(cur);
      if (p) for (const x of p) stack.push(x);
    }
    return false;
  };

  const operandParts = (op) => {
    const s = new Set();
    for (let i = 0; i < op.numOperands; i++) {
      const d = op.getOperand(i).definingOp;
      if (!d) continue;
      const part = opToPart.get(d);
      if (part) s.add(part);
    }
    return s;
  };

  const recordEdges = (op, own) => {
    for (const part of operandParts(op)) {
      if (part === own) continue;
      let p = preds.get(own);
      if (!p) { p = new Set(); preds.set(own, p); }
      p.add(part);
    }
  };

  for (const op of topo) {
    const label = labelOf(op);
    if (label == null) continue;
    let merged = false;
    for (let i = 0; i < op.numOperands; i++) {
      const producer = op.getOperand(i).definingOp;
      if (!producer) continue;
      const pPart = opToPart.get(producer);
      if (!pPart || !sameLabel(pPart.label, label)) continue;
      if (!canMerge(pPart, op, label)) continue;

      let createsCycle = false;
      for (const part of operandParts(op)) {
        if (part === pPart) continue;
        if (isUpstreamOf(pPart, part)) { createsCycle = true; break; }
      }
      if (createsCycle) continue;

      pPart.ops.push(op);
      pPart.opSet.add(op);
      onAttach(pPart, op);
      opToPart.set(op, pPart);
      recordEdges(op, pPart);
      merged = true;
      break;
    }
    if (!merged) {
      const part = { id: nextId++, label, ops: [op], opSet: new Set([op]) };
      onAttach(part, op);
      partitions.push(part);
      opToPart.set(op, part);
      recordEdges(op, part);
    }
  }

  return { partitions, opToPart, preds };
}

export function computePartitionIO(opSet, iterOps, { pullConstants = false, isConstant = () => false } = {}) {
  const inputs = [], inputSet = new Set();
  const outputs = [], outputSet = new Set();
  const constDefs = [], constSet = new Set();

  for (const op of iterOps) {
    for (let i = 0; i < op.numOperands; i++) {
      const v = op.getOperand(i);
      const d = v.definingOp;
      if (d && opSet.has(d)) continue;
      if (pullConstants && d && isConstant(d)) {
        if (!constSet.has(d)) { constSet.add(d); constDefs.push(d); }
        continue;
      }
      if (!inputSet.has(v)) { inputSet.add(v); inputs.push(v); }
    }
    for (let i = 0; i < op.numResults; i++) {
      const r = op.getResult(i);
      if (outputSet.has(r)) continue;
      let escapes = false;
      for (const use of r.uses()) {
        if (!opSet.has(use.user)) { escapes = true; break; }
      }
      if (escapes) { outputSet.add(r); outputs.push(r); }
    }
  }

  return { inputs, outputs, constDefs };
}

export function topoSortPartitions(partitions, preds) {
  const inDeg = new Map();
  const adj = new Map();
  for (const p of partitions) { inDeg.set(p, 0); adj.set(p, []); }
  for (const p of partitions) {
    const ps = preds.get(p);
    if (!ps) continue;
    for (const q of ps) {
      if (!adj.has(q)) continue;
      adj.get(q).push(p);
      inDeg.set(p, inDeg.get(p) + 1);
    }
  }
  const queue = [];
  for (const p of partitions) if (inDeg.get(p) === 0) queue.push(p);
  const out = [];
  let head = 0;
  while (head < queue.length) {
    const p = queue[head++];
    out.push(p);
    for (const c of adj.get(p)) {
      const d = inDeg.get(c) - 1;
      inDeg.set(c, d);
      if (d === 0) queue.push(c);
    }
  }
  return out.length === partitions.length ? out : null;
}
