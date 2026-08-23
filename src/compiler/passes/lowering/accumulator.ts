import { SeqNode, LetStmtNode } from '../../ir/tensor/nodes.js';
import { walk as irWalk } from '../../ir/ir_visitor.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { BlockNode, BufferLoadNode, BufferStoreNode, ForNode, IntImmNode, MathOpNode, TirNode, VariableNode } from '../../ir/tensor/nodes.js';

export type AccumulatorInfo = {
  store: BufferStoreNode;
  loadSide: BufferLoadNode;
  valueSide: TirNode;
  outerIndices: TirNode[];
  block: BlockNode;
  op: string;
  prologue: TirNode | null;
};

export const ACCUMULATOR_OPS = new Set(['+', '*', 'max', 'min']);

function splitAccumulatorBody(body: TirNode): { store: BufferStoreNode; prologue: TirNode | null } | null {
  const lets: LetStmtNode[] = [];
  let node: TirNode = body;
  while (node && node.type === 'LetStmtNode') {
    lets.push(node as LetStmtNode);
    node = (node as LetStmtNode).body;
  }

  let store: BufferStoreNode;
  let rest: TirNode[] = [];
  if (node && node.type === 'BufferStoreNode') {
    store = node as BufferStoreNode;
  } else if (node && node.type === 'SeqNode') {
    const stmts = (node as SeqNode).stmts;
    const last = stmts[stmts.length - 1];
    if (!last || last.type !== 'BufferStoreNode') return null;
    store = last as BufferStoreNode;
    rest = stmts.slice(0, -1);
  } else {
    return null;
  }

  if (lets.length === 0 && rest.length === 0) return { store, prologue: null };

  let prologue: TirNode = rest.length === 1 ? rest[0] : new SeqNode(rest);
  for (let i = lets.length - 1; i >= 0; i--) {
    prologue = new LetStmtNode(lets[i].variable, lets[i].value, prologue);
  }
  return { store, prologue };
}

function touchesBuffer(root: TirNode, buffer: Buffer): boolean {
  let found = false;
  irWalk(root as unknown as IRNode, ((node: TirNode) => {
    if (node.type !== 'BufferLoadNode' && node.type !== 'BufferStoreNode') return;
    if ((node as unknown as { buffer: Buffer }).buffer === buffer) found = true;
  }) as never);
  return found;
}

export function detectAccumulator(forNode: ForNode): AccumulatorInfo | null {
  const block = forNode.body as BlockNode;
  if (!block || block.type !== 'BlockNode') return null;

  const parts = splitAccumulatorBody(block.body);
  if (!parts) return null;
  const { store, prologue } = parts;
  if (prologue && touchesBuffer(prologue, store.buffer)) return null;

  const val = store.value as MathOpNode;
  if (!val || val.type !== 'MathOpNode' || !ACCUMULATOR_OPS.has(val.op)) return null;

  let loadSide: BufferLoadNode | null = null;
  let valueSide: TirNode | null = null;

  if (val.a && val.a.type === 'BufferLoadNode' && (val.a as BufferLoadNode).buffer === store.buffer) {
    loadSide = val.a as BufferLoadNode;
    valueSide = val.b;
  } else if (val.b && val.b.type === 'BufferLoadNode' && (val.b as BufferLoadNode).buffer === store.buffer) {
    loadSide = val.b as BufferLoadNode;
    valueSide = val.a;
  }
  if (!loadSide) return null;

  const storeKey = indicesKey(store.indices);
  const loadKey = indicesKey(loadSide.indices);
  if (storeKey !== loadKey) return null;
  if (storeKey.includes('?')) return null;

  const outerIndices: TirNode[] = store.indices.map((idx: TirNode) => {
    if (idx.type !== 'VariableNode') return idx;
    for (const bind of block.iterVars) {
      if (bind.iterVar && bind.iterVar.name === (idx as VariableNode).name && bind.binding) {
        return bind.binding;
      }
    }
    return idx;
  });

  const loopVarName = forNode.loopVar.name;
  const resolvedKey = indicesKey(outerIndices);
  if (resolvedKey.includes('?')) return null;
  if (resolvedKey.includes('$' + loopVarName)) return null;

  return {
    store,
    loadSide,
    valueSide: valueSide as TirNode,
    outerIndices,
    block,
    op: (store.value as MathOpNode).op,
    prologue,
  };
}

function indicesKey(indices: readonly TirNode[]): string {
  return indices.map(exprKey).join(',');
}

function exprKey(node: TirNode | null | undefined): string {
  if (!node) return '?';
  if (node.type === 'VariableNode') return '$' + (node as VariableNode).name;
  if (node.type === 'IntImmNode') return String((node as IntImmNode).value);
  if (node.type === 'MathOpNode') {
    const m = node as MathOpNode;
    return '(' + exprKey(m.a) + m.op + (m.b ? exprKey(m.b) : '') + ')';
  }
  return '?';
}
