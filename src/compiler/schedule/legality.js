import { Analyzer } from '../analysis/analyzer.js';
import { SymInt } from '../analysis/sym_int.js';
import { irBound, analyzerForLoops } from '../analysis/ir_arith.js';
import { carriesDependence, permutationPreservesDependences } from '../analysis/dependence.js';
import { IterVarKind } from '../ir/tensor/nodes.js';

export { analyzerForLoops };

export const IterVarPolicy = Object.freeze({
  SPATIAL: new Set([IterVarKind.DATA_PAR]),
  REORDERABLE: new Set([IterVarKind.DATA_PAR, IterVarKind.COMM_REDUCE]),
  ACCUMULABLE: new Set([IterVarKind.DATA_PAR, IterVarKind.COMM_REDUCE]),
});

function blockAbstractionPermits(state, enclosingLoop, loopVarNames, allowedKinds, byBlock) {
  const sref = state.tree.getSRef(enclosingLoop);
  if (!sref) return false;
  const blocks = sref.childBlocks();
  if (blocks.length === 0) return false;
  for (const blockSRef of blocks) {
    const info = byBlock.get(blockSRef.node);
    if (!info) return false;
    for (const name of loopVarNames) {
      const kinds = info.iterKindsOfLoopVar(name);
      if (!kinds) return false;
      for (const kind of kinds) if (!allowedKinds.has(kind)) return false;
    }
  }
  return true;
}

export function loopCarriedDependence(state, loop, allowedKinds) {
  const { info, deps } = state.nestAnalysis(loop);
  const dep = carriesDependence(deps, loop);
  if (!dep) return null;
  if (blockAbstractionPermits(state, loop, [loop.loopVar.name], allowedKinds, info.byBlock)) return null;
  return `loop '${loop.loopVar.name}' carries a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}

export function reorderLegality(state, chain, after) {
  const permuted = chain.filter((loop, i) => loop !== after[i]);
  if (permuted.length === 0) return null;
  const names = permuted.map((l) => l.loopVar.name);
  const { info, deps } = state.nestAnalysis(chain[0]);
  if (blockAbstractionPermits(state, chain[chain.length - 1], names, IterVarPolicy.REORDERABLE, info.byBlock)) return null;
  const dep = permutationPreservesDependences(deps, chain, after);
  if (!dep) return null;
  return `permutation violates a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}

export function classifyBufferIndex(analyzer, indexExpr, dimExtent) {
  if (typeof dimExtent !== 'number' || dimExtent < 0) return 'unknown';
  const b = irBound(analyzer, indexExpr);
  if (b === null) return 'unknown';
  if (b.min >= 0 && b.max <= dimExtent - 1) return 'in';
  if (b.min > dimExtent - 1 || b.max < 0) return 'oob';
  return 'unknown';
}

export function proveDivisible(extent, factor) {
  if (!Number.isInteger(factor) || factor <= 0) return false;
  const b = new Analyzer().constIntBound(SymInt.mod(extent, factor));
  return b.min === 0 && b.max === 0;
}

export function collectVarsUsed(node, out) {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'VariableNode':
      if (node.name) out.add(node.name);
      return;
    case 'BufferLoadNode':
      if (node.indices) for (const idx of node.indices) collectVarsUsed(idx, out);
      return;
    case 'BufferStoreNode':
      if (node.indices) for (const idx of node.indices) collectVarsUsed(idx, out);
      collectVarsUsed(node.value, out);
      return;
    case 'MathOpNode':
      collectVarsUsed(node.a, out);
      if (node.b) collectVarsUsed(node.b, out);
      return;
    case 'CompareNode':
      collectVarsUsed(node.a, out);
      collectVarsUsed(node.b, out);
      return;
    case 'CastNode':
      collectVarsUsed(node.expr, out);
      return;
    case 'CallExternNode':
      for (const a of node.args) collectVarsUsed(a, out);
      return;
    case 'IfThenElseNode':
      collectVarsUsed(node.condition, out);
      collectVarsUsed(node.thenBody, out);
      if (node.elseBody) collectVarsUsed(node.elseBody, out);
      return;
    case 'SeqNode':
      for (const s of node.stmts) collectVarsUsed(s, out);
      return;
    case 'LetStmtNode':
      collectVarsUsed(node.value, out);
      collectVarsUsed(node.body, out);
      return;
    default:
      return;
  }
}

export function collectWriteIndexVars(block, out) {
  const writeBufs = new Set((block.writes || []).map(w => w.buffer && w.buffer.name));
  const stack = [block.body, block.initBody];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    if (n.type === 'BufferStoreNode' && n.buffer && writeBufs.has(n.buffer.name)) {
      if (n.indices) for (const idx of n.indices) collectVarsUsed(idx, out);
    }
    if (n.type === 'BufferStoreNode') { if (n.value) stack.push(n.value); continue; }
    if (n.type === 'SeqNode') { for (const s of n.stmts) stack.push(s); continue; }
    if (n.type === 'IfThenElseNode') { stack.push(n.thenBody); if (n.elseBody) stack.push(n.elseBody); continue; }
    if (n.type === 'LetStmtNode') { stack.push(n.body); continue; }
  }
}

export function reductionLoopVars(block) {
  const writeIdx = new Set();
  collectWriteIndexVars(block, writeIdx);
  const used = new Set();
  collectVarsUsed(block.body, used);
  if (block.initBody) collectVarsUsed(block.initBody, used);

  const own = new Set();
  const spatial = new Set();
  const reduction = new Set();
  for (const iv of block.iterVars || []) {
    if (!iv || !iv.iterVar) continue;
    own.add(iv.iterVar.name);
    const bindVars = new Set();
    collectVarsUsed(iv.binding, bindVars);
    const isSpatial = writeIdx.has(iv.iterVar.name) || [...bindVars].some((v) => writeIdx.has(v));
    for (const v of bindVars) (isSpatial ? spatial : reduction).add(v);
  }
  for (const name of used) {
    if (!own.has(name) && !writeIdx.has(name)) reduction.add(name);
  }
  for (const name of spatial) reduction.delete(name);
  return reduction;
}

export function collectBlocksUnder(node, out) {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'BlockNode':
      out.push(node);
      collectBlocksUnder(node.body, out);
      if (node.initBody) collectBlocksUnder(node.initBody, out);
      return;
    case 'ForNode':
    case 'AllocateNode':
    case 'LetStmtNode':
      collectBlocksUnder(node.body, out);
      return;
    case 'SeqNode':
      for (const s of node.stmts) collectBlocksUnder(s, out);
      return;
    case 'IfThenElseNode':
      collectBlocksUnder(node.thenBody, out);
      if (node.elseBody) collectBlocksUnder(node.elseBody, out);
      return;
    default:
      return;
  }
}

