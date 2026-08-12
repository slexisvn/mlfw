import { classifyBlock } from '../schedule/rules.js';
import type { NodeSlots } from '../ir/tensor/nodes.js';
import type { TirNode, PrimFunc, BlockNode } from '../ir/tensor/nodes.js';


export type BlockStructure = { spatial: number; reduction: number; reads: number; hasReduction: boolean };

function* walkStmts(root: TirNode | null | undefined): Generator<TirNode, void, undefined> {
  const stack: (TirNode | null | undefined)[] = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) continue;
    yield node;
    const slots = node as unknown as NodeSlots;
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.stmts) for (const st of slots.stmts as TirNode[]) stack.push(st);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    if (slots.initBody) stack.push(slots.initBody as TirNode);
  }
}

export function findBlock(root: TirNode | null | undefined, name: string): BlockNode | null {
  for (const node of walkStmts(root)) {
    if (node.type === 'BlockNode' && (node as BlockNode).name === name) return node as BlockNode;
  }
  return null;
}

export function collectAllBlockNames(root: TirNode | null | undefined): string[] {
  const names: string[] = [];
  for (const node of walkStmts(root)) {
    if (node.type === 'BlockNode') names.push((node as BlockNode).name);
  }
  return names;
}

export function analyzeBlockStructure(primFunc: PrimFunc, blockName: string): BlockStructure {
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
