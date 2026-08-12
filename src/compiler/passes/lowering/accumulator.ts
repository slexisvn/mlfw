import type { BlockNode, BufferLoadNode, BufferStoreNode, ForNode, IntImmNode, MathOpNode, TirNode, VariableNode } from '../../ir/tensor/nodes.js';

export type AccumulatorInfo = {
  store: BufferStoreNode;
  loadSide: BufferLoadNode;
  valueSide: TirNode;
  outerIndices: TirNode[];
  block: BlockNode;
  op: string;
};

export const ACCUMULATOR_OPS = new Set(['+', '*', 'max', 'min']);

export function detectAccumulator(forNode: ForNode): AccumulatorInfo | null {
  const block = forNode.body as BlockNode;
  if (!block || block.type !== 'BlockNode') return null;

  const inner = block.body;
  if (!inner || inner.type !== 'BufferStoreNode') return null;

  const store = inner as BufferStoreNode;
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
