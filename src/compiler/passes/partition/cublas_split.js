import { GraphFunction } from '../../ir/graph/function.js';
import { Operation } from '../../ir/graph/operation.js';

const CONSTANT_OPS = new Set(['constant', 'scalar_constant']);
const TERMINATORS = new Set(['return', 'yield']);

function isConstantOp(op) {
  return CONSTANT_OPS.has(op.opName);
}

const PARTITION_BUFFER_LIMIT = 32 * 1024;

function maxResultBytes(op) {
  let m = 0;
  for (let i = 0; i < op.numResults; i++) {
    const t = op.getResult(i).type;
    if (!t || !t.isFullyStatic) continue;
    const b = t.sizeInBytes();
    if (b > m) m = b;
  }
  return m;
}

function constScalarOf(v) {
  let op = v.definingOp;
  if (op && op.opName === 'broadcast') {
    const src = op.getOperand(0);
    op = src && src.definingOp;
  }
  if (op && isConstantOp(op)) {
    const val = op.getAttr('value');
    if (typeof val === 'number') return val;
  }
  return 0;
}

function cublasDotInfo(op) {
  if (op.opName !== 'dot') return null;
  const lhsT = op.getOperand(0).type;
  const rhsT = op.getOperand(1).type;
  const outT = op.getResult(0).type;
  if (!lhsT || !rhsT || !outT) return null;
  if (lhsT.dtype !== 'f32' || rhsT.dtype !== 'f32' || outT.dtype !== 'f32') return null;
  const lhsBatch = op.getAttr('lhs_batch') || [];
  const rhsBatch = op.getAttr('rhs_batch') || [];
  if (lhsBatch.length > 0 || rhsBatch.length > 0) return null;
  const lhsC = op.getAttr('lhs_contracting') || [];
  const rhsC = op.getAttr('rhs_contracting') || [];
  if (lhsC.length !== 1 || rhsC.length !== 1) return null;
  if (rhsT.rank !== 2) return null;
  if (lhsC[0] !== lhsT.rank - 1) return null;
  if (rhsC[0] !== 0 && rhsC[0] !== 1) return null;
  if (!lhsT.isFullyStatic || !rhsT.isFullyStatic || !outT.isFullyStatic) return null;
  const lhsDef = op.getOperand(0).definingOp;
  const rhsDef = op.getOperand(1).definingOp;
  if (lhsDef && isConstantOp(lhsDef)) return null;
  if (rhsDef && isConstantOp(rhsDef)) return null;
  const transB = rhsC[0] === 1;
  const K = lhsT.shape[lhsT.rank - 1];
  const rhsK = transB ? rhsT.shape[1] : rhsT.shape[0];
  if (rhsK !== K) return null;
  let M = 1;
  for (let i = 0; i < lhsT.rank - 1; i++) M *= lhsT.shape[i];
  const N = transB ? rhsT.shape[0] : rhsT.shape[1];
  if (M <= 0 || N <= 0 || K <= 0) return null;
  return { M, N, K, transB };
}

function topoSortOps(ops) {
  const opSet = new Set(ops);
  const ordered = [];
  const state = new Map();
  const visit = (op) => {
    const s = state.get(op);
    if (s === 1 || s === 2) return;
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

function buildPartitions(partitionOps, opTarget) {
  const topo = topoSortOps(partitionOps);
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
    const target = opTarget.get(op);
    const opBytes = maxResultBytes(op);
    let merged = false;
    for (let i = 0; i < op.numOperands; i++) {
      const producer = op.getOperand(i).definingOp;
      if (!producer) continue;
      const pPart = opToPart.get(producer);
      if (!pPart || pPart.target !== target) continue;

      if (Math.max(pPart.maxBuf || 0, opBytes) > PARTITION_BUFFER_LIMIT) continue;

      let createsCycle = false;
      for (const part of operandParts(op)) {
        if (part === pPart) continue;
        if (isUpstreamOf(pPart, part)) { createsCycle = true; break; }
      }
      if (createsCycle) continue;

      pPart.ops.push(op);
      pPart.opSet.add(op);
      pPart.maxBuf = Math.max(pPart.maxBuf || 0, opBytes);
      opToPart.set(op, pPart);
      recordEdges(op, pPart);
      merged = true;
      break;
    }
    if (!merged) {
      const part = { id: nextId++, target, ops: [op], opSet: new Set([op]), maxBuf: opBytes };
      partitions.push(part);
      opToPart.set(op, part);
      recordEdges(op, part);
    }
  }

  return { partitions, opToPart, preds };
}

function topoSortPartitions(partitions, preds) {
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
  while (queue.length > 0) {
    const p = queue.shift();
    out.push(p);
    for (const c of adj.get(p)) {
      const d = inDeg.get(c) - 1;
      inDeg.set(c, d);
      if (d === 0) queue.push(c);
    }
  }
  return out.length === partitions.length ? out : null;
}

function materializePartition(part, name, dotInfoMap) {
  const opSet = part.opSet;
  const sorted = topoSortOps(part.ops);
  const inputs = [], inputSet = new Set();
  const outputs = [], outputSet = new Set();
  const constDefs = [], constSet = new Set();

  for (const op of sorted) {
    for (let i = 0; i < op.numOperands; i++) {
      const v = op.getOperand(i);
      const d = v.definingOp;
      if (d && opSet.has(d)) continue;
      if (d && isConstantOp(d)) {
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

  for (const v of inputs) if (!v.type || !v.type.isFullyStatic) return null;
  for (const v of outputs) if (!v.type || !v.type.isFullyStatic) return null;

  const subFunc = new GraphFunction(name, inputs.map(v => v.type), outputs.map(v => v.type));
  const valueMap = new Map();
  for (let i = 0; i < inputs.length; i++) valueMap.set(inputs[i], subFunc.args[i]);

  for (const c of constDefs) {
    subFunc.entryBlock.pushOp(c.clone(valueMap));
  }
  for (const op of sorted) {
    subFunc.entryBlock.pushOp(op.clone(valueMap));
  }

  const retOperands = outputs.map(v => valueMap.get(v));
  if (retOperands.some(v => v === undefined)) return null;
  subFunc.entryBlock.pushOp(new Operation('return', retOperands, []));

  const dotOp = (part.ops.length === 1 && dotInfoMap.has(part.ops[0])) ? part.ops[0] : null;
  return { part, subFunc, inputs, outputs, dotOp };
}

const BOUNDARY_OP_NAMES = new Set(['dot', 'reduce']);

function containsBoundaryOp(op) {
  if (BOUNDARY_OP_NAMES.has(op.opName)) return true;
  if (op.regions) {
    for (const region of op.regions) {
      const block = region.entryBlock;
      if (!block) continue;
      for (const inner of block.ops()) {
        if (containsBoundaryOp(inner)) return true;
      }
    }
  }
  return false;
}

function buildExecutionPlan(func, retOp, built) {
  const slotOf = new Map();
  let nextSlot = 0;
  const getSlot = (v) => {
    let s = slotOf.get(v);
    if (s === undefined) { s = nextSlot++; slotOf.set(v, s); }
    return s;
  };

  for (const arg of func.args) getSlot(arg);
  for (const b of built) for (const v of b.outputs) getSlot(v);

  const argSlots = [];
  for (const arg of func.args) argSlots.push(getSlot(arg));

  const returnFixups = [];
  const usedRetSlot = new Set();
  for (let i = 0; i < retOp.numOperands; i++) {
    const v = retOp.getOperand(i);
    const pos = argSlots.length;
    const isBlockArg = v.isBlockArgument && v.isBlockArgument();
    const isConst = v.definingOp && isConstantOp(v.definingOp);

    if (!isBlockArg && !isConst && slotOf.has(v)) {
      const s = slotOf.get(v);
      if (!usedRetSlot.has(s)) {
        usedRetSlot.add(s);
        argSlots.push(s);
        continue;
      }
      argSlots.push(nextSlot++);
      returnFixups.push({ pos, kind: 'copy', srcSlot: s });
      continue;
    }

    if (isBlockArg) {
      argSlots.push(nextSlot++);
      returnFixups.push({ pos, kind: 'copy', srcSlot: getSlot(v) });
      continue;
    }

    if (isConst) {
      argSlots.push(nextSlot++);
      returnFixups.push({ pos, kind: 'const', value: constScalarOf(v) });
      continue;
    }

    return null;
  }

  const steps = [];
  for (const b of built) {
    const inputSlots = [];
    for (const v of b.inputs) {
      const s = slotOf.get(v);
      if (s === undefined) return null;
      inputSlots.push(s);
    }
    const outputSlots = b.outputs.map(v => slotOf.get(v));
    steps.push({ name: b.subFunc.name, inputSlots, outputSlots });
  }

  const argSlotSet = new Set(argSlots);
  const intermediates = [];
  const seenSlot = new Set();
  for (const [v, s] of slotOf) {
    if (argSlotSet.has(s) || seenSlot.has(s)) continue;
    seenSlot.add(s);
    if (!v.type || !v.type.isFullyStatic) return null;
    intermediates.push({ slot: s, shape: [...v.type.shape], dtype: v.type.dtype });
  }

  return { plan: { numSlots: nextSlot, argSlots, intermediates, steps, returnFixups } };
}

export function splitGraphForNative(graphModule) {
  if (graphModule.functionCount !== 1) return null;
  const func = graphModule.functions().next().value;
  const retOp = func.getReturnOp();
  if (!retOp) return null;

  const partitionOps = [];
  const opTarget = new Map();
  let boundaryCount = 0;

  for (const op of func.ops()) {
    if (TERMINATORS.has(op.opName)) continue;
    if (isConstantOp(op)) continue;
    if (containsBoundaryOp(op)) opTarget.set(op, 'boundary#' + boundaryCount++);
    else opTarget.set(op, 'native');
    partitionOps.push(op);
  }

  if (boundaryCount < 2 || partitionOps.length === 0) return null;

  const { partitions, preds } = buildPartitions(partitionOps, opTarget);
  if (partitions.length < 2) return null;

  const orderedParts = topoSortPartitions(partitions, preds);
  if (!orderedParts) return null;

  const baseName = func.name;
  const built = [];
  const emptyDotInfo = new Map();
  let idx = 0;
  for (const part of orderedParts) {
    const m = materializePartition(part, baseName + '_p' + (idx++), emptyDotInfo);
    if (!m) return null;
    built.push(m);
  }

  const planResult = buildExecutionPlan(func, retOp, built);
  if (!planResult) return null;

  graphModule.removeFunction(func.name);
  for (const b of built) graphModule.addFunction(b.subFunc);

  return { plan: planResult.plan };
}

export function splitGraphForCublas(graphModule) {
  if (graphModule.functionCount !== 1) return null;
  const func = graphModule.functions().next().value;
  const retOp = func.getReturnOp();
  if (!retOp) return null;

  const partitionOps = [];
  const opTarget = new Map();
  const dotInfoMap = new Map();
  let dotCount = 0;

  for (const op of func.ops()) {
    if (TERMINATORS.has(op.opName)) continue;
    if (isConstantOp(op)) continue;
    const info = cublasDotInfo(op);
    if (info) {
      opTarget.set(op, 'cublas#' + dotCount);
      dotInfoMap.set(op, info);
      dotCount++;
    } else {
      opTarget.set(op, 'native');
    }
    partitionOps.push(op);
  }

  if (dotCount === 0 || partitionOps.length === 0) return null;

  const { partitions, preds } = buildPartitions(partitionOps, opTarget);
  if (partitions.length < 2) return null;

  const orderedParts = topoSortPartitions(partitions, preds);
  if (!orderedParts) return null;

  const baseName = func.name;
  const built = [];
  let idx = 0;
  for (const part of orderedParts) {
    const m = materializePartition(part, baseName + '_p' + (idx++), dotInfoMap);
    if (!m) return null;
    built.push(m);
  }

  const planResult = buildExecutionPlan(func, retOp, built);
  if (!planResult) return null;

  const cublasInfos = new Map();
  for (const b of built) {
    if (!b.dotOp) continue;
    const info = dotInfoMap.get(b.dotOp);
    const aIdx = b.inputs.indexOf(b.dotOp.getOperand(0));
    const bIdx = b.inputs.indexOf(b.dotOp.getOperand(1));
    const cPos = b.outputs.indexOf(b.dotOp.getResult(0));
    if (aIdx < 0 || bIdx < 0 || cPos < 0) continue;
    cublasInfos.set(b.subFunc.name, {
      M: info.M, N: info.N, K: info.K, transB: info.transB,
      aIdx, bIdx, cIdx: b.inputs.length + cPos,
    });
  }

  if (cublasInfos.size === 0) return null;

  graphModule.removeFunction(func.name);
  for (const b of built) graphModule.addFunction(b.subFunc);

  return { plan: planResult.plan, cublasInfos };
}
