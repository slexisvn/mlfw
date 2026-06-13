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

export function loopVarIsReductionOf(varName, block) {
  const used = new Set();
  collectVarsUsed(block.body, used);
  if (block.initBody) collectVarsUsed(block.initBody, used);
  if (!used.has(varName)) return false;
  const written = new Set();
  collectWriteIndexVars(block, written);
  return !written.has(varName);
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

export function loopCarriesReduction(loop) {
  const varName = loop.loopVar.name;
  const blocks = [];
  collectBlocksUnder(loop.body, blocks);
  for (const block of blocks) {
    if (loopVarIsReductionOf(varName, block)) return block.name;
  }
  return null;
}
