export const ACCUMULATOR_OPS = new Set(['+', '*', 'max', 'min']);

export function detectAccumulator(forNode) {
  const block = forNode.body;
  if (!block || block.type !== 'BlockNode') return null;

  const inner = block.body;
  if (!inner || inner.type !== 'BufferStoreNode') return null;

  const store = inner;
  const val = store.value;
  if (!val || val.type !== 'MathOpNode' || !ACCUMULATOR_OPS.has(val.op)) return null;

  let loadSide = null;
  let valueSide = null;

  if (val.a && val.a.type === 'BufferLoadNode' && val.a.buffer === store.buffer) {
    loadSide = val.a;
    valueSide = val.b;
  } else if (val.b && val.b.type === 'BufferLoadNode' && val.b.buffer === store.buffer) {
    loadSide = val.b;
    valueSide = val.a;
  }
  if (!loadSide) return null;

  const storeKey = indicesKey(store.indices);
  const loadKey = indicesKey(loadSide.indices);
  if (storeKey !== loadKey) return null;
  if (storeKey.includes('?')) return null;

  const outerIndices = store.indices.map(idx => {
    if (idx.type !== 'VariableNode') return idx;
    for (const bind of block.iterVars) {
      if (bind.iterVar && bind.iterVar.name === idx.name && bind.binding) {
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
    valueSide,
    outerIndices,
    block,
    op: store.value.op,
  };
}

function indicesKey(indices) {
  return indices.map(exprKey).join(',');
}

function exprKey(node) {
  if (!node) return '?';
  if (node.type === 'VariableNode') return '$' + node.name;
  if (node.type === 'IntImmNode') return String(node.value);
  if (node.type === 'MathOpNode') {
    return '(' + exprKey(node.a) + node.op + (node.b ? exprKey(node.b) : '') + ')';
  }
  return '?';
}
