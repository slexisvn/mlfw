import { Analyzer } from '../analysis/analyzer.js';
import { SymInt } from '../analysis/sym_int.js';
import { irBound, analyzerForLoops } from '../analysis/ir_arith.js';
import { carriesDependence, permutationPreservesDependences } from '../analysis/dependence.js';
import { IterVarKind } from '../ir/tensor/nodes.js';
import type { TirNode, ForNode, BlockNode, SeqNode, IfThenElseNode, LetStmtNode, AllocateNode, VariableNode, BufferLoadNode, BufferStoreNode, MathOpNode, CompareNode, CastNode, CallExternNode } from '../ir/tensor/nodes.js';
import type { Analyzer as AnalyzerType } from '../analysis/analyzer.js';
import type { ScheduleState } from './schedule_state.js';
import { isPrivateToLoop } from '../analysis/buffer_access.js';
import type { BlockAccessInfo } from '../analysis/buffer_access.js';
import type { SymExpr } from '../analysis/sym_int.js';

export type IndexClassification = 'in' | 'oob' | 'unknown';

export { analyzerForLoops };

export const IterVarPolicy = Object.freeze({
  SPATIAL: new Set([IterVarKind.DATA_PAR]),
  REORDERABLE: new Set([IterVarKind.DATA_PAR, IterVarKind.COMM_REDUCE]),
  ACCUMULABLE: new Set([IterVarKind.DATA_PAR, IterVarKind.COMM_REDUCE]),
});

function blockAbstractionPermits(state: ScheduleState, enclosingLoop: TirNode, loopVarNames: readonly string[], allowedKinds: ReadonlySet<string>, byBlock: ReadonlyMap<BlockNode, BlockAccessInfo>): boolean {
  const sref = state.tree.getSRef(enclosingLoop);
  if (!sref) return false;
  const blocks = sref.childBlocks();
  if (blocks.length === 0) return false;
  for (const blockSRef of blocks) {
    const info = byBlock.get(blockSRef.node as BlockNode);
    if (!info) return false;
    for (const name of loopVarNames) {
      const kinds = info.iterKindsOfLoopVar(name);
      if (!kinds) return false;
      for (const kind of kinds) if (!allowedKinds.has(kind as string)) return false;
    }
  }
  return true;
}

export function loopCarriedDependence(state: ScheduleState, loop: ForNode, allowedKinds: ReadonlySet<string>): string | null {
  const { info, deps } = state.nestAnalysis(loop);
  const dep = carriesDependence(deps, loop, (buffer) => isPrivateToLoop(info, buffer, loop));
  if (!dep) return null;
  if (blockAbstractionPermits(state, loop, [loop.loopVar.name], allowedKinds, info.byBlock)) return null;
  return `loop '${loop.loopVar.name}' carries a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}

export function reorderLegality(state: ScheduleState, chain: readonly ForNode[], after: readonly ForNode[]): string | null {
  const permuted = chain.filter((loop, i) => loop !== after[i]);
  if (permuted.length === 0) return null;
  const names = permuted.map((l) => l.loopVar.name);
  const { info, deps } = state.nestAnalysis(chain[0]);
  if (blockAbstractionPermits(state, chain[chain.length - 1], names, IterVarPolicy.REORDERABLE, info.byBlock)) return null;
  const dep = permutationPreservesDependences(deps, chain, after);
  if (!dep) return null;
  return `permutation violates a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}

export function classifyBufferIndex(analyzer: AnalyzerType, indexExpr: TirNode, dimExtent: unknown): IndexClassification {
  if (typeof dimExtent !== 'number' || dimExtent < 0) return 'unknown';
  const b = irBound(analyzer, indexExpr);
  if (b === null) return 'unknown';
  if (b.min >= 0 && b.max <= dimExtent - 1) return 'in';
  if (b.min > dimExtent - 1 || b.max < 0) return 'oob';
  return 'unknown';
}

export function proveDivisible(extent: SymExpr, factor: number): boolean {
  if (!Number.isInteger(factor) || factor <= 0) return false;
  const b = new Analyzer().constIntBound(SymInt.mod(extent, factor));
  return b.min === 0 && b.max === 0;
}

export function collectVarsUsed(node: TirNode | null | undefined, out: Set<string>): void {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'VariableNode': {
      const v = node as VariableNode;
      if (v.name) out.add(v.name);
      return;
    }
    case 'BufferLoadNode': {
      const l = node as BufferLoadNode;
      if (l.indices) for (const idx of l.indices) collectVarsUsed(idx, out);
      return;
    }
    case 'BufferStoreNode': {
      const st = node as BufferStoreNode;
      if (st.indices) for (const idx of st.indices) collectVarsUsed(idx, out);
      collectVarsUsed(st.value, out);
      return;
    }
    case 'MathOpNode': {
      const m = node as MathOpNode;
      collectVarsUsed(m.a, out);
      if (m.b) collectVarsUsed(m.b, out);
      return;
    }
    case 'CompareNode': {
      const c = node as CompareNode;
      collectVarsUsed(c.a, out);
      collectVarsUsed(c.b, out);
      return;
    }
    case 'CastNode':
      collectVarsUsed((node as CastNode).expr, out);
      return;
    case 'CallExternNode':
      for (const a of (node as CallExternNode).args) collectVarsUsed(a, out);
      return;
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      collectVarsUsed(ite.condition, out);
      collectVarsUsed(ite.thenBody, out);
      if (ite.elseBody) collectVarsUsed(ite.elseBody, out);
      return;
    }
    case 'SeqNode':
      for (const st of (node as SeqNode).stmts) collectVarsUsed(st, out);
      return;
    case 'LetStmtNode': {
      const ls = node as LetStmtNode;
      collectVarsUsed(ls.value, out);
      collectVarsUsed(ls.body, out);
      return;
    }
    default:
      return;
  }
}

export function collectWriteIndexVars(block: BlockNode, out: Set<string>): void {
  const writeBufs = new Set((block.writes || []).map(w => w.buffer && w.buffer.name));
  const stack: (TirNode | null | undefined)[] = [block.body, block.initBody];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    const st = n as BufferStoreNode;
    if (n.type === 'BufferStoreNode' && st.buffer && writeBufs.has(st.buffer.name)) {
      if (st.indices) for (const idx of st.indices) collectVarsUsed(idx, out);
    }
    if (n.type === 'BufferStoreNode') { if (st.value) stack.push(st.value); continue; }
    if (n.type === 'SeqNode') { for (const x of (n as SeqNode).stmts) stack.push(x); continue; }
    if (n.type === 'IfThenElseNode') { const ite = n as IfThenElseNode; stack.push(ite.thenBody); if (ite.elseBody) stack.push(ite.elseBody); continue; }
    if (n.type === 'LetStmtNode') { stack.push((n as LetStmtNode).body); continue; }
  }
}

function collectLetBoundVars(node: TirNode | null | undefined, out: Set<string>): void {
  const stack: (TirNode | null | undefined)[] = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    if (n.type === 'LetStmtNode') {
      const ls = n as LetStmtNode;
      if (ls.variable) out.add(ls.variable.name);
      stack.push(ls.body);
      continue;
    }
    if (n.type === 'SeqNode') { for (const x of (n as SeqNode).stmts) stack.push(x); continue; }
    if (n.type === 'IfThenElseNode') { const ite = n as IfThenElseNode; stack.push(ite.thenBody); if (ite.elseBody) stack.push(ite.elseBody); continue; }
  }
}

/**
 * Loop vars that carry a reduction, so a schedule may not run their iterations in parallel.
 * A declared CommReduce axis is authoritative; the write-index heuristic only classifies axes
 * the block did not label, and cannot see past a block that writes at more than one rank.
 */
export function reductionLoopVars(block: BlockNode): Set<string> {
  const writeIdx = new Set<string>();
  collectWriteIndexVars(block, writeIdx);
  const used = new Set<string>();
  collectVarsUsed(block.body, used);
  if (block.initBody) collectVarsUsed(block.initBody, used);

  const letBound = new Set<string>();
  collectLetBoundVars(block.body, letBound);
  collectLetBoundVars(block.initBody, letBound);

  const own = new Set<string>();
  const spatial = new Set<string>();
  const reduction = new Set<string>();
  for (const iv of block.iterVars || []) {
    if (!iv || !iv.iterVar) continue;
    own.add(iv.iterVar.name);
    const bindVars = new Set<string>();
    collectVarsUsed(iv.binding, bindVars);
    const isSpatial = iv.kind !== IterVarKind.COMM_REDUCE
      && (writeIdx.has(iv.iterVar.name) || [...bindVars].some((v) => writeIdx.has(v)));
    for (const v of bindVars) (isSpatial ? spatial : reduction).add(v);
  }
  for (const name of used) {
    if (own.has(name) || letBound.has(name) || writeIdx.has(name)) continue;
    reduction.add(name);
  }
  for (const name of spatial) reduction.delete(name);
  return reduction;
}

export function collectBlocksUnder(node: TirNode | null | undefined, out: BlockNode[]): void {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'BlockNode': {
      const b = node as BlockNode;
      out.push(b);
      collectBlocksUnder(b.body, out);
      if (b.initBody) collectBlocksUnder(b.initBody, out);
      return;
    }
    case 'ForNode':
    case 'AllocateNode':
    case 'LetStmtNode':
      collectBlocksUnder((node as ForNode | AllocateNode | LetStmtNode).body, out);
      return;
    case 'SeqNode':
      for (const st of (node as SeqNode).stmts) collectBlocksUnder(st, out);
      return;
    case 'IfThenElseNode': {
      const ite = node as IfThenElseNode;
      collectBlocksUnder(ite.thenBody, out);
      if (ite.elseBody) collectBlocksUnder(ite.elseBody, out);
      return;
    }
    default:
      return;
  }
}

