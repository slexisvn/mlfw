import { GraphFunction } from '../../ir/graph/function.js';
import { Operation } from '../../ir/graph/operation.js';
import { topoSortOps, buildPartitions, topoSortPartitions, computePartitionIO } from './partition_core.js';
import { isConstantOp as isConstantOpName, isTerminatorOp, containsLaunchBoundary } from '../../ir/graph/op_traits.js';
import type { GraphModule } from '../../ir/graph/module.js';
import type { Block } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { TensorType } from '../../ir/graph/types.js';
import type { BuildPartitionsOpts, Partition } from './partition_core.js';

export type CublasDotInfo = { M: number; N: number; K: number; transB: boolean };
type BufferedPartition = Partition & { maxBuf?: number };
export type MaterializedPartition = {
  part: Partition;
  subFunc: GraphFunction;
  inputs: Value[];
  outputs: Value[];
  dotOp: Operation | null;
};
type ReturnFixup = { pos: number; kind: string; srcSlot?: number; value?: number };
type PlanStep = { name: string; inputSlots: number[]; outputSlots: number[] };
type PlanIntermediate = { slot: number; shape: readonly number[]; dtype: string };
export type ExecutionPlan = {
  numSlots: number;
  argSlots: number[];
  intermediates: PlanIntermediate[];
  steps: PlanStep[];
  returnFixups: ReturnFixup[];
};
export type CublasKernelInfo = { M: number; N: number; K: number; transB: boolean; aIdx: number; bIdx: number; cIdx: number };

export function isConstantOp(op: Operation): boolean {
  return isConstantOpName(op.opName);
}

const PARTITION_BUFFER_LIMIT = 32 * 1024;

function maxResultBytes(op: Operation): number {
  let m = 0;
  for (let i = 0; i < op.numResults; i++) {
    const t = op.getResult(i).type as TensorType;
    if (!t || !t.isFullyStatic) continue;
    const b = t.sizeInBytes() as number;
    if (b > m) m = b;
  }
  return m;
}

function constScalarOf(v: Value): number {
  let op = v.definingOp;
  if (op && op.opName === 'broadcast') {
    const src = op.getOperand(0);
    op = src && src.definingOp;
  }
  if (op && isConstantOp(op)) {
    const val = op.getAttr<number>('value');
    if (typeof val === 'number') return val;
  }
  return 0;
}

export function cublasDotInfo(op: Operation): CublasDotInfo | null {
  if (op.opName !== 'dot' && op.opName !== 'cublas_gemm') return null;
  const lhsT = op.getOperand(0).type as TensorType;
  const rhsT = op.getOperand(1).type as TensorType;
  const outT = op.getResult(0).type as TensorType;
  if (!lhsT || !rhsT || !outT) return null;
  if (lhsT.dtype !== 'f32' || rhsT.dtype !== 'f32' || outT.dtype !== 'f32') return null;
  const lhsBatch = op.getAttr<readonly number[]>('lhs_batch') || [];
  const rhsBatch = op.getAttr<readonly number[]>('rhs_batch') || [];
  if (lhsBatch.length > 0 || rhsBatch.length > 0) return null;
  const lhsC = op.getAttr<readonly number[]>('lhs_contracting') || [];
  const rhsC = op.getAttr<readonly number[]>('rhs_contracting') || [];
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
  const K = lhsT.shape[lhsT.rank - 1] as number;
  const rhsK = (transB ? rhsT.shape[1] : rhsT.shape[0]) as number;
  if (rhsK !== K) return null;
  let M = 1;
  for (let i = 0; i < lhsT.rank - 1; i++) M *= lhsT.shape[i] as number;
  const N = (transB ? rhsT.shape[0] : rhsT.shape[1]) as number;
  if (M <= 0 || N <= 0 || K <= 0) return null;
  return { M, N, K, transB };
}

export { topoSortOps };

function bufferLimitedConfig(opTarget: ReadonlyMap<Operation, string>): BuildPartitionsOpts {
  return {
    labelOf: (op: Operation) => opTarget.get(op),
    canMerge: (part: Partition, op: Operation) => Math.max((part as BufferedPartition).maxBuf || 0, maxResultBytes(op)) <= PARTITION_BUFFER_LIMIT,
    onAttach: (part: Partition, op: Operation) => { (part as BufferedPartition).maxBuf = Math.max((part as BufferedPartition).maxBuf || 0, maxResultBytes(op)); },
  };
}

export function materializePartition(part: Partition, name: string, dotInfoMap: ReadonlyMap<Operation, CublasDotInfo>): MaterializedPartition | null {
  const opSet = part.opSet;
  const sorted = topoSortOps(part.ops);
  const { inputs, outputs, constDefs } = computePartitionIO(opSet, sorted, { pullConstants: true, isConstant: isConstantOp });

  for (const v of inputs) if (!v.type || !(v.type as TensorType).isFullyStatic) return null;
  for (const v of outputs) if (!v.type || !(v.type as TensorType).isFullyStatic) return null;

  const subFunc = new GraphFunction(name, inputs.map(v => v.type), outputs.map(v => v.type));
  const valueMap = new Map<Value, Value>();
  for (let i = 0; i < inputs.length; i++) valueMap.set(inputs[i], subFunc.args[i]);

  const entry = subFunc.entryBlock as Block;
  for (const c of constDefs) {
    entry.pushOp(c.clone(valueMap));
  }
  for (const op of sorted) {
    entry.pushOp(op.clone(valueMap));
  }

  const retOperands = outputs.map(v => valueMap.get(v));
  if (retOperands.some(v => v === undefined)) return null;
  entry.pushOp(new Operation('return', retOperands as Value[], []));

  const dotOp = (part.ops.length === 1 && dotInfoMap.has(part.ops[0])) ? part.ops[0] : null;
  return { part, subFunc, inputs, outputs, dotOp };
}

function maxBoundaryResultBytes(op: Operation): number {
  let m = 0;
  for (let i = 0; i < op.numResults; i++) {
    const t = op.getResult(i).type as TensorType;
    if (!t || !t.isFullyStatic) continue;
    let n = 1;
    for (const d of t.shape) n *= d as number;
    if (n > m) m = n;
  }
  return m;
}

export function hasDependentBoundaries(graphModule: GraphModule, minIntermediate = 256): boolean {
  if (graphModule.functionCount !== 1) return false;
  const func = graphModule.functions().next().value as GraphFunction;
  const memo = new Map<Operation, number>();
  const maxBoundaryInSubtree = (op: Operation | null | undefined): number => {
    if (!op) return 0;
    const cached = memo.get(op);
    if (cached !== undefined) return cached;
    memo.set(op, 0);
    let m = containsLaunchBoundary(op) ? maxBoundaryResultBytes(op) : 0;
    for (let i = 0; i < op.numOperands; i++) {
      const u = maxBoundaryInSubtree(op.getOperand(i).definingOp);
      if (u > m) m = u;
    }
    memo.set(op, m);
    return m;
  };
  for (const op of func.ops()) {
    if (isTerminatorOp(op.opName) || !containsLaunchBoundary(op)) continue;
    for (let i = 0; i < op.numOperands; i++) {
      if (maxBoundaryInSubtree(op.getOperand(i).definingOp) > minIntermediate) return true;
    }
  }
  return false;
}

function buildExecutionPlan(func: GraphFunction, retOp: Operation, built: readonly MaterializedPartition[]): { plan: ExecutionPlan } | null {
  const slotOf = new Map<Value, number>();
  let nextSlot = 0;
  const getSlot = (v: Value): number => {
    let s = slotOf.get(v);
    if (s === undefined) { s = nextSlot++; slotOf.set(v, s); }
    return s;
  };

  for (const arg of func.args) getSlot(arg);
  for (const b of built) for (const v of b.outputs) getSlot(v);

  const argSlots: number[] = [];
  for (const arg of func.args) argSlots.push(getSlot(arg));

  const returnFixups: ReturnFixup[] = [];
  const usedRetSlot = new Set<number>();
  for (let i = 0; i < retOp.numOperands; i++) {
    const v = retOp.getOperand(i);
    const pos = argSlots.length;
    const isBlockArg = v.isBlockArgument && v.isBlockArgument();
    const isConst = v.definingOp && isConstantOp(v.definingOp);

    if (!isBlockArg && !isConst && slotOf.has(v)) {
      const s = slotOf.get(v) as number;
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

  const steps: PlanStep[] = [];
  for (const b of built) {
    const inputSlots: number[] = [];
    for (const v of b.inputs) {
      const s = slotOf.get(v);
      if (s === undefined) return null;
      inputSlots.push(s);
    }
    const outputSlots = b.outputs.map(v => slotOf.get(v) as number);
    steps.push({ name: b.subFunc.name, inputSlots, outputSlots });
  }

  const argSlotSet = new Set<number>(argSlots);
  const intermediates: PlanIntermediate[] = [];
  const seenSlot = new Set<number>();
  for (const [v, s] of slotOf) {
    if (argSlotSet.has(s) || seenSlot.has(s)) continue;
    seenSlot.add(s);
    const vt = v.type as TensorType;
    if (!vt || !vt.isFullyStatic) return null;
    intermediates.push({ slot: s, shape: [...vt.shape] as number[], dtype: vt.dtype });
  }

  return { plan: { numSlots: nextSlot, argSlots, intermediates, steps, returnFixups } };
}

export function splitGraphForNative(graphModule: GraphModule, minBoundaries = 2): { plan: ExecutionPlan } | null {
  if (graphModule.functionCount !== 1) return null;
  const func = graphModule.functions().next().value as GraphFunction;
  const retOp = func.getReturnOp();
  if (!retOp) return null;

  const partitionOps: Operation[] = [];
  const opTarget = new Map<Operation, string>();
  let boundaryCount = 0;

  for (const op of func.ops()) {
    if (isTerminatorOp(op.opName)) continue;
    if (isConstantOp(op)) continue;
    if (containsLaunchBoundary(op)) opTarget.set(op, 'boundary#' + boundaryCount++);
    else opTarget.set(op, 'native');
    partitionOps.push(op);
  }

  if (boundaryCount < minBoundaries || partitionOps.length === 0) return null;

  const { partitions, preds } = buildPartitions(partitionOps, bufferLimitedConfig(opTarget));
  if (partitions.length < 2) return null;

  const orderedParts = topoSortPartitions(partitions, preds);
  if (!orderedParts) return null;

  const baseName = func.name;
  const built: MaterializedPartition[] = [];
  const emptyDotInfo = new Map<Operation, CublasDotInfo>();
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

export function splitGraphForCublas(graphModule: GraphModule): { plan: ExecutionPlan; cublasInfos: Map<string, CublasKernelInfo> } | null {
  if (graphModule.functionCount !== 1) return null;
  const func = graphModule.functions().next().value as GraphFunction;
  const retOp = func.getReturnOp();
  if (!retOp) return null;

  const partitionOps: Operation[] = [];
  const opTarget = new Map<Operation, string>();
  const dotInfoMap = new Map<Operation, CublasDotInfo>();
  let dotCount = 0;
  let boundaryCount = 0;

  for (const op of func.ops()) {
    if (isTerminatorOp(op.opName)) continue;
    if (isConstantOp(op)) continue;
    const info = cublasDotInfo(op);
    if (info) {
      opTarget.set(op, 'cublas#' + dotCount);
      dotInfoMap.set(op, info);
      dotCount++;
    } else if (containsLaunchBoundary(op)) {
      opTarget.set(op, 'boundary#' + boundaryCount++);
    } else {
      opTarget.set(op, 'native');
    }
    partitionOps.push(op);
  }

  if (dotCount + boundaryCount === 0 || partitionOps.length === 0) return null;

  const { partitions, preds } = buildPartitions(partitionOps, bufferLimitedConfig(opTarget));
  if (partitions.length < 2) return null;

  const orderedParts = topoSortPartitions(partitions, preds);
  if (!orderedParts) return null;

  const baseName = func.name;
  const built: MaterializedPartition[] = [];
  let idx = 0;
  for (const part of orderedParts) {
    const m = materializePartition(part, baseName + '_p' + (idx++), dotInfoMap);
    if (!m) return null;
    built.push(m);
  }

  const planResult = buildExecutionPlan(func, retOp, built);
  if (!planResult) return null;

  const cublasInfos = new Map<string, CublasKernelInfo>();
  for (const b of built) {
    if (!b.dotOp) continue;
    const info = dotInfoMap.get(b.dotOp) as CublasDotInfo;
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
