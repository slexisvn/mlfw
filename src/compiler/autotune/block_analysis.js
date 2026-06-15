import { classifyBlock } from '../schedule/rules.js';

function* walkStmts(root) {
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) continue;
    yield node;
    if (node.body) stack.push(node.body);
    if (node.stmts) for (const s of node.stmts) stack.push(s);
    if (node.thenBody) stack.push(node.thenBody);
    if (node.elseBody) stack.push(node.elseBody);
    if (node.initBody) stack.push(node.initBody);
  }
}

export function findBlock(root, name) {
  for (const node of walkStmts(root)) {
    if (node.type === 'BlockNode' && node.name === name) return node;
  }
  return null;
}

export function collectAllBlockNames(root) {
  const names = [];
  for (const node of walkStmts(root)) {
    if (node.type === 'BlockNode') names.push(node.name);
  }
  return names;
}

export function analyzeBlockStructure(primFunc, blockName) {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return { spatial: 0, reduction: 0, reads: 0, hasReduction: false };
  let spatial = 0;
  let reduction = 0;
  for (const l of info.loops) {
    if (info.reductionLoopVars.has(l.loopVar.name)) reduction++;
    else spatial++;
  }
  return { spatial, reduction, reads: info.readBuffers.length, hasReduction: info.hasReduction };
}
