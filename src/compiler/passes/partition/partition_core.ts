import { readValues, topoSortOpSet } from '../../ir/graph/graph_algorithms.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';

export type PartitionLabel = unknown;
export type Partition = { id: number; label: PartitionLabel; ops: Operation[]; opSet: Set<Operation> };
export type BuildPartitionsOpts = {
  labelOf(op: Operation): PartitionLabel;
  sameLabel?(a: PartitionLabel, b: PartitionLabel): boolean;
  canMerge?(part: Partition, op: Operation, label: PartitionLabel): boolean;
  onAttach?(part: Partition, op: Operation): void;
  sort?(ops: readonly Operation[]): Operation[];
};
export type PartitionGraph = {
  partitions: Partition[];
  opToPart: Map<Operation, Partition>;
  preds: Map<Partition, Set<Partition>>;
};
export type PartitionIO = { inputs: Value[]; outputs: Value[]; constDefs: Operation[] };
export type PartitionIOOpts = { pullConstants?: boolean; isConstant?(op: Operation): boolean };

export function topoSortOps(ops: readonly Operation[]): Operation[] {
  return topoSortOpSet(ops, 'throw') as Operation[];
}

export function buildPartitions(partitionOps: readonly Operation[], { labelOf, sameLabel = (a, b) => a === b, canMerge = () => true, onAttach = () => {}, sort = topoSortOps }: BuildPartitionsOpts): PartitionGraph {
  const topo = sort(partitionOps);
  const opToPart = new Map<Operation, Partition>();
  const preds = new Map<Partition, Set<Partition>>();
  const partitions: Partition[] = [];
  let nextId = 0;

  const isUpstreamOf = (ancestor: Partition, node: Partition): boolean => {
    if (ancestor === node) return true;
    const stack: Partition[] = [node];
    const seen = new Set<Partition>();
    while (stack.length > 0) {
      const cur = stack.pop() as Partition;
      if (cur === ancestor) return true;
      if (seen.has(cur)) continue;
      seen.add(cur);
      const p = preds.get(cur);
      if (p) for (const x of p) stack.push(x);
    }
    return false;
  };

  const operandParts = (op: Operation): Set<Partition> => {
    const s = new Set<Partition>();
    for (const v of readValues(op)) {
      const d = v.definingOp;
      if (!d) continue;
      const part = opToPart.get(d);
      if (part) s.add(part);
    }
    return s;
  };

  const recordEdges = (op: Operation, own: Partition): void => {
    for (const part of operandParts(op)) {
      if (part === own) continue;
      let p = preds.get(own);
      if (!p) { p = new Set<Partition>(); preds.set(own, p); }
      p.add(part);
    }
  };

  for (const op of topo) {
    const label = labelOf(op);
    if (label == null) continue;
    let merged = false;
    for (const v of readValues(op)) {
      const producer = v.definingOp;
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
      const part: Partition = { id: nextId++, label, ops: [op], opSet: new Set([op]) };
      onAttach(part, op);
      partitions.push(part);
      opToPart.set(op, part);
      recordEdges(op, part);
    }
  }

  return { partitions, opToPart, preds };
}

export function computePartitionIO(opSet: ReadonlySet<Operation>, iterOps: Iterable<Operation>, { pullConstants = false, isConstant = () => false }: PartitionIOOpts = {}): PartitionIO {
  const inputs: Value[] = [], inputSet = new Set<Value>();
  const outputs: Value[] = [], outputSet = new Set<Value>();
  const constDefs: Operation[] = [], constSet = new Set<Operation>();

  for (const op of iterOps) {
    for (const v of readValues(op)) {
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

export function topoSortPartitions(partitions: readonly Partition[], preds: ReadonlyMap<Partition, Set<Partition>>): Partition[] | null {
  const inDeg = new Map<Partition, number>();
  const adj = new Map<Partition, Partition[]>();
  for (const p of partitions) { inDeg.set(p, 0); adj.set(p, []); }
  for (const p of partitions) {
    const ps = preds.get(p);
    if (!ps) continue;
    for (const q of ps) {
      if (!adj.has(q)) continue;
      (adj.get(q) as Partition[]).push(p);
      inDeg.set(p, (inDeg.get(p) as number) + 1);
    }
  }
  const queue: Partition[] = [];
  for (const p of partitions) if (inDeg.get(p) === 0) queue.push(p);
  const out: Partition[] = [];
  let head = 0;
  while (head < queue.length) {
    const p = queue[head++];
    out.push(p);
    for (const c of adj.get(p) as Partition[]) {
      const d = (inDeg.get(c) as number) - 1;
      inDeg.set(c, d);
      if (d === 0) queue.push(c);
    }
  }
  return out.length === partitions.length ? out : null;
}
