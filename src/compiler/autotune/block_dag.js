function walkBlocks(root, visit) {
  const stack = [root];
  while (stack.length) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    if (n.type === 'BlockNode') visit(n);
    if (n.body) stack.push(n.body);
    if (n.initBody) stack.push(n.initBody);
    if (n.thenBody) stack.push(n.thenBody);
    if (n.elseBody) stack.push(n.elseBody);
    if (Array.isArray(n.stmts)) for (const s of n.stmts) stack.push(s);
  }
}

export function buildBlockDAG(primFunc) {
  const blocks = [];
  walkBlocks(primFunc.body, (b) => {
    blocks.push({
      name: b.name,
      node: b,
      reads: (b.reads || []).map(r => r.buffer && r.buffer.name).filter(Boolean),
      writes: (b.writes || []).map(w => w.buffer && w.buffer.name).filter(Boolean)
    });
  });
  const consumersByBuf = new Map();
  for (const b of blocks) {
    for (const r of b.reads) {
      if (!consumersByBuf.has(r)) consumersByBuf.set(r, []);
      consumersByBuf.get(r).push(b.name);
    }
  }
  return { blocks, consumersByBuf, byName: new Map(blocks.map(b => [b.name, b])) };
}

function indexVarNames(indices) {
  const names = [];
  for (const idx of indices) {
    if (!idx || idx.type !== 'VariableNode') return null;
    names.push(idx.name);
  }
  return names;
}

function collectLoads(node, bufName, acc) {
  if (!node || typeof node !== 'object') return acc;
  if (node.type === 'BufferLoadNode' && node.buffer && node.buffer.name === bufName) acc.push(node);
  for (const key of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (node[key]) collectLoads(node[key], bufName, acc);
  }
  if (Array.isArray(node.args)) for (const a of node.args) collectLoads(a, bufName, acc);
  if (Array.isArray(node.indices)) for (const a of node.indices) collectLoads(a, bufName, acc);
  if (Array.isArray(node.stmts)) for (const a of node.stmts) collectLoads(a, bufName, acc);
  return acc;
}

function collectLoadedBuffers(node, acc) {
  if (!node || typeof node !== 'object') return acc;
  if (node.type === 'BufferLoadNode' && node.buffer && node.buffer.name) acc.add(node.buffer.name);
  for (const key of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (node[key]) collectLoadedBuffers(node[key], acc);
  }
  if (Array.isArray(node.args)) for (const a of node.args) collectLoadedBuffers(a, acc);
  if (Array.isArray(node.indices)) for (const a of node.indices) collectLoadedBuffers(a, acc);
  if (Array.isArray(node.stmts)) for (const a of node.stmts) collectLoadedBuffers(a, acc);
  return acc;
}

export function findFusibleConsumer(primFunc, dag, producerBlockName, classify) {
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
  const pLoaded = new Set(producer.reads);
  collectLoadedBuffers(producer.node.body, pLoaded);
  collectLoadedBuffers(producer.node.initBody, pLoaded);
  if (pLoaded.has(cBlk.writes[0])) return null;

  const pStore = producer.node.body && producer.node.body.type === 'BufferStoreNode' ? producer.node.body : null;
  const cStore = cBlk.node.body && cBlk.node.body.type === 'BufferStoreNode' ? cBlk.node.body : null;
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
