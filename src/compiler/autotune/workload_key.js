export function buildBlockMap(root) {
  const map = new Map();
  const stack = [root];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BlockNode') map.set(n.name, n);
    if (n.body) stack.push(n.body);
    if (n.stmts) for (const s of n.stmts) stack.push(s);
    if (n.thenBody) stack.push(n.thenBody);
    if (n.elseBody) stack.push(n.elseBody);
    if (n.initBody) stack.push(n.initBody);
  }
  return map;
}

export function computeWorkloadKey(primFunc, blockName, target, blockMap = null) {
  const map = blockMap || buildBlockMap(primFunc.body);
  const block = map.get(blockName) || null;
  const parts = [blockName];

  if (block) {
    const bufShapes = [];
    for (const r of block.reads) bufShapes.push(`${r.buffer.shape.join('x')}:${r.buffer.dtype}`);
    for (const w of block.writes) bufShapes.push(`${w.buffer.shape.join('x')}:${w.buffer.dtype}`);
    parts.push(bufShapes.join(','));

    const ops = [];
    collectBlockOps(block.body, ops);
    if (block.initBody) collectBlockOps(block.initBody, ops);
    parts.push(ops.join(';'));
  }

  parts.push(target.name);
  parts.push(target.kind);

  return fnv1a(parts.join('|'));
}

function collectBlockOps(node, ops) {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'BufferStoreNode':
      ops.push('store');
      if (node.indices) for (const idx of node.indices) collectBlockOps(idx, ops);
      collectBlockOps(node.value, ops);
      return;
    case 'BufferLoadNode':
      ops.push(`load:${node.buffer.name}`);
      if (node.indices) for (const idx of node.indices) collectBlockOps(idx, ops);
      return;
    case 'MathOpNode':
      ops.push(`math:${node.op}`);
      collectBlockOps(node.a, ops);
      if (node.b) collectBlockOps(node.b, ops);
      return;
    case 'CallExternNode':
      ops.push(`call:${node.externName}`);
      for (const arg of node.args) collectBlockOps(arg, ops);
      return;
    case 'CompareNode':
      ops.push(`cmp:${node.direction}`);
      collectBlockOps(node.a, ops);
      collectBlockOps(node.b, ops);
      return;
    case 'CastNode':
      ops.push(`cast:${node.fromDtype}->${node.toDtype}`);
      collectBlockOps(node.expr, ops);
      return;
    case 'IfThenElseNode':
      collectBlockOps(node.condition, ops);
      collectBlockOps(node.thenBody, ops);
      if (node.elseBody) collectBlockOps(node.elseBody, ops);
      return;
    case 'SeqNode':
      for (const s of node.stmts) collectBlockOps(s, ops);
      return;
    case 'ForNode':
      collectBlockOps(node.body, ops);
      return;
    case 'BlockNode':
      if (node.initBody) collectBlockOps(node.initBody, ops);
      collectBlockOps(node.body, ops);
      return;
    case 'LetStmtNode':
      collectBlockOps(node.value, ops);
      collectBlockOps(node.body, ops);
      return;
    default:
      return;
  }
}

function fnv1a(str) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}
