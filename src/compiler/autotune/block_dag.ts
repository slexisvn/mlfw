import type { NodeSlots } from '../ir/tensor/nodes.js';
import type { TirNode, PrimFunc, BlockNode, BufferLoadNode, BufferStoreNode, VariableNode } from '../ir/tensor/nodes.js';
import type { BlockClassification } from '../schedule/rules.js';


export type DagBlock = {
  name: string;
  node: BlockNode;
  reads: string[];
  writes: string[];
};

export type BlockDAG = {
  blocks: DagBlock[];
  consumersByBuf: Map<string, string[]>;
  byName: Map<string, DagBlock>;
};

export type ClassifyFn = (primFunc: PrimFunc, blockName: string) => BlockClassification | null;

function walkBlocks(root: TirNode | null | undefined, visit: (b: BlockNode) => void): void {
  const stack: (TirNode | null | undefined)[] = [root];
  while (stack.length) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    if (n.type === 'BlockNode') visit(n as BlockNode);
    const slots = n as unknown as NodeSlots;
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.initBody) stack.push(slots.initBody as TirNode);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
    if (Array.isArray(slots.stmts)) for (const st of slots.stmts as TirNode[]) stack.push(st);
  }
}

export function buildBlockDAG(primFunc: PrimFunc): BlockDAG {
  const blocks: DagBlock[] = [];
  walkBlocks(primFunc.body, (b: BlockNode) => {
    blocks.push({
      name: b.name,
      node: b,
      reads: (b.reads || []).map(r => r.buffer && r.buffer.name).filter((x): x is string => !!x),
      writes: (b.writes || []).map(w => w.buffer && w.buffer.name).filter((x): x is string => !!x)
    });
  });
  const consumersByBuf = new Map<string, string[]>();
  for (const b of blocks) {
    for (const r of b.reads) {
      if (!consumersByBuf.has(r)) consumersByBuf.set(r, []);
      (consumersByBuf.get(r) as string[]).push(b.name);
    }
  }
  return { blocks, consumersByBuf, byName: new Map(blocks.map(b => [b.name, b])) };
}

function indexVarNames(indices: readonly TirNode[]): string[] | null {
  const names: string[] = [];
  for (const idx of indices) {
    if (!idx || idx.type !== 'VariableNode') return null;
    names.push((idx as VariableNode).name);
  }
  return names;
}

function collectLoads(node: TirNode | null | undefined, bufName: string, acc: BufferLoadNode[]): BufferLoadNode[] {
  if (!node || typeof node !== 'object') return acc;
  const ld = node as BufferLoadNode;
  if (node.type === 'BufferLoadNode' && ld.buffer && ld.buffer.name === bufName) acc.push(ld);
  const slots = node as unknown as NodeSlots;
  for (const key of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (slots[key]) collectLoads(slots[key] as TirNode, bufName, acc);
  }
  if (Array.isArray(slots.args)) for (const a of slots.args as TirNode[]) collectLoads(a, bufName, acc);
  if (Array.isArray(slots.indices)) for (const a of slots.indices as TirNode[]) collectLoads(a, bufName, acc);
  if (Array.isArray(slots.stmts)) for (const a of slots.stmts as TirNode[]) collectLoads(a, bufName, acc);
  return acc;
}

function collectLoadedBuffers(node: TirNode | null | undefined, acc: Set<string>): Set<string> {
  if (!node || typeof node !== 'object') return acc;
  const ld = node as BufferLoadNode;
  if (node.type === 'BufferLoadNode' && ld.buffer && ld.buffer.name) acc.add(ld.buffer.name);
  const slots = node as unknown as NodeSlots;
  for (const key of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (slots[key]) collectLoadedBuffers(slots[key] as TirNode, acc);
  }
  if (Array.isArray(slots.args)) for (const a of slots.args as TirNode[]) collectLoadedBuffers(a, acc);
  if (Array.isArray(slots.indices)) for (const a of slots.indices as TirNode[]) collectLoadedBuffers(a, acc);
  if (Array.isArray(slots.stmts)) for (const a of slots.stmts as TirNode[]) collectLoadedBuffers(a, acc);
  return acc;
}

export function findFusibleConsumer(primFunc: PrimFunc, dag: BlockDAG, producerBlockName: string, classify: ClassifyFn): string | null {
  const producer = dag.byName.get(producerBlockName);
  if (!producer || producer.writes.length !== 1) return null;
  const out = producer.writes[0];

  const consumers = [...new Set((dag.consumersByBuf.get(out) || []).filter(n => n !== producerBlockName))];
  if (consumers.length !== 1) return null;
  const cName = consumers[0];

  const pInfo = classify(primFunc, producerBlockName);
  const cInfo = classify(primFunc, cName);
  if (!pInfo || !cInfo || cInfo.hasReduction) return null;

  const cBlk = dag.byName.get(cName);
  if (!cBlk || cBlk.writes.length !== 1 || cBlk.writes[0] === out) return null;
  const pLoaded = new Set<string>(producer.reads);
  collectLoadedBuffers(producer.node.body, pLoaded);
  collectLoadedBuffers(producer.node.initBody, pLoaded);
  if (pLoaded.has(cBlk.writes[0])) return null;

  const pStore = producer.node.body && producer.node.body.type === 'BufferStoreNode' ? producer.node.body as BufferStoreNode : null;
  const cStore = cBlk.node.body && cBlk.node.body.type === 'BufferStoreNode' ? cBlk.node.body as BufferStoreNode : null;
  if (!pStore || !cStore) return null;

  const pSpatialNames = pInfo.loops.filter(l => !pInfo.reductionLoopVars.has(l.loopVar.name)).map(l => l.loopVar.name);
  const pWriteNames = indexVarNames(pStore.indices);
  if (!pWriteNames || pWriteNames.join(',') !== pSpatialNames.join(',')) return null;

  const cLoopNames = cInfo.loops.map(l => l.loopVar.name);
  if (cLoopNames.length !== pSpatialNames.length) return null;
  const cWriteNames = indexVarNames(cStore.indices);
  if (!cWriteNames || cWriteNames.join(',') !== cLoopNames.join(',')) return null;

  const tLoads = collectLoads(cStore.value, out, []);
  if (tLoads.length === 0) return null;
  for (const ld of tLoads) {
    const rn = indexVarNames(ld.indices);
    if (!rn || rn.join(',') !== cWriteNames.join(',')) return null;
  }
  return cName;
}
