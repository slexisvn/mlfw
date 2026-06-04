export function computeWorkloadKey(primFunc, blockName, target) {
  const parts = [];

  parts.push(blockName);

  const paramShapes = [];
  for (const [, buf] of primFunc.bufferMap) {
    paramShapes.push(`${buf.name}:${buf.shape.join('x')}:${buf.dtype}`);
  }
  parts.push(paramShapes.join(','));

  const ops = [];
  collectOpSignature(primFunc.body, blockName, ops);
  parts.push(ops.join(';'));

  parts.push(target.name);
  parts.push(target.kind);

  return fnv1a(parts.join('|'));
}

function collectOpSignature(node, targetBlock, ops) {
  if (!node) return false;
  switch (node.type) {
    case 'BlockNode':
      if (node.name === targetBlock) {
        collectBlockOps(node.body, ops);
        if (node.initBody) collectBlockOps(node.initBody, ops);
        return true;
      }
      if (node.initBody && collectOpSignature(node.initBody, targetBlock, ops)) return true;
      return collectOpSignature(node.body, targetBlock, ops);
    case 'ForNode':
      return collectOpSignature(node.body, targetBlock, ops);
    case 'SeqNode':
      for (const s of node.stmts) {
        if (collectOpSignature(s, targetBlock, ops)) return true;
      }
      return false;
    case 'IfThenElseNode':
      if (collectOpSignature(node.thenBody, targetBlock, ops)) return true;
      if (node.elseBody) return collectOpSignature(node.elseBody, targetBlock, ops);
      return false;
    case 'AllocateNode':
      return collectOpSignature(node.body, targetBlock, ops);
    case 'LetStmtNode':
      return collectOpSignature(node.body, targetBlock, ops);
    default:
      return false;
  }
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
      ops.push('load');
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
