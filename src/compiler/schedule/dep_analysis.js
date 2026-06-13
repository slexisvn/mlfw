export const DepKind = Object.freeze({
  RAW: 'read-after-write',
  WAR: 'write-after-read',
  WAW: 'write-after-write',
});

export class Dependency {
  constructor(src, dst, kind, bufferName) {
    this.src = src;
    this.dst = dst;
    this.kind = kind;
    this.bufferName = bufferName;
  }
}

function collectBufferAccesses(blockNode) {
  const reads = new Set();
  const writes = new Set();
  visitExprTree(blockNode.body, reads, writes);
  if (blockNode.initBody) visitExprTree(blockNode.initBody, reads, writes);
  return { reads, writes };
}

function visitExprTree(node, reads, writes) {
  if (!node || typeof node !== 'object') return;
  switch (node.type) {
    case 'BufferLoadNode':
      if (node.buffer) reads.add(node.buffer.name);
      if (node.indices) for (const idx of node.indices) visitExprTree(idx, reads, writes);
      return;
    case 'BufferStoreNode':
      if (node.buffer) writes.add(node.buffer.name);
      if (node.indices) for (const idx of node.indices) visitExprTree(idx, reads, writes);
      visitExprTree(node.value, reads, writes);
      return;
    case 'MathOpNode':
      visitExprTree(node.a, reads, writes);
      if (node.b) visitExprTree(node.b, reads, writes);
      return;
    case 'CompareNode':
      visitExprTree(node.a, reads, writes);
      visitExprTree(node.b, reads, writes);
      return;
    case 'CallExternNode':
      for (const arg of node.args) visitExprTree(arg, reads, writes);
      return;
    case 'IfThenElseNode':
      visitExprTree(node.condition, reads, writes);
      visitExprTree(node.thenBody, reads, writes);
      if (node.elseBody) visitExprTree(node.elseBody, reads, writes);
      return;
    case 'CastNode':
      visitExprTree(node.expr, reads, writes);
      return;
    case 'LetStmtNode':
      visitExprTree(node.value, reads, writes);
      visitExprTree(node.body, reads, writes);
      return;
    case 'SeqNode':
      for (const s of node.stmts) visitExprTree(s, reads, writes);
      return;
    case 'ForNode':
      visitExprTree(node.body, reads, writes);
      return;
    case 'BlockNode':
      if (node.initBody) visitExprTree(node.initBody, reads, writes);
      visitExprTree(node.body, reads, writes);
      return;
    default:
      return;
  }
}

export class DependencyAnalysis {
  constructor(srefTree) {
    this._tree = srefTree;
    this._blockAccesses = new Map();
    for (const blockSRef of srefTree.allBlocks()) {
      this._blockAccesses.set(blockSRef.node.name, collectBufferAccesses(blockSRef.node));
    }
  }

  getReads(blockName) {
    const acc = this._blockAccesses.get(blockName);
    return acc ? [...acc.reads] : [];
  }

  getWrites(blockName) {
    const acc = this._blockAccesses.get(blockName);
    return acc ? [...acc.writes] : [];
  }

  computeDeps(srcBlock, dstBlock) {
    const srcAcc = this._blockAccesses.get(srcBlock);
    const dstAcc = this._blockAccesses.get(dstBlock);
    if (!srcAcc || !dstAcc) return [];

    const deps = [];

    for (const buf of srcAcc.writes) {
      if (dstAcc.reads.has(buf)) deps.push(new Dependency(srcBlock, dstBlock, DepKind.RAW, buf));
      if (dstAcc.writes.has(buf)) deps.push(new Dependency(srcBlock, dstBlock, DepKind.WAW, buf));
    }

    for (const buf of srcAcc.reads) {
      if (dstAcc.writes.has(buf)) deps.push(new Dependency(srcBlock, dstBlock, DepKind.WAR, buf));
    }

    return deps;
  }

  canReorder(blockA, blockB) {
    const srcAcc = this._blockAccesses.get(blockA);
    const dstAcc = this._blockAccesses.get(blockB);
    if (!srcAcc || !dstAcc) return true;
    for (const buf of srcAcc.writes) {
      if (dstAcc.reads.has(buf) || dstAcc.writes.has(buf)) return false;
    }
    for (const buf of srcAcc.reads) {
      if (dstAcc.writes.has(buf)) return false;
    }
    return true;
  }
}
