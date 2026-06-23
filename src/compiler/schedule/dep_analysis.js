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
  return { reads, writes, accesses: collectRegions(blockNode) };
}

function indexRanges(indices, loops) {
  if (!indices) return null;
  return indices.map((idx) => {
    if (idx && idx.type === 'VariableNode' && loops.has(idx.name)) {
      const info = loops.get(idx.name);
      return info.ext === null ? null : [info.min, info.ext];
    }
    if (idx && idx.type === 'IntImmNode') return [idx.value, 1];
    return null;
  });
}

function collectRegions(blockNode) {
  const accesses = [];
  const loops = new Map();
  const walk = (node) => {
    if (!node || typeof node !== 'object') return;
    switch (node.type) {
      case 'ForNode': {
        const e = node.extent;
        const ext = e && e.type === 'IntImmNode' ? e.value : (typeof e === 'number' ? e : null);
        const m = node.min;
        const min = m && m.type === 'IntImmNode' ? m.value : (typeof m === 'number' ? m : 0);
        if (node.loopVar) loops.set(node.loopVar.name, { min, ext });
        walk(node.body);
        return;
      }
      case 'BufferLoadNode':
        if (node.buffer) accesses.push({ name: node.buffer.name, kind: 'read', ranges: indexRanges(node.indices, loops) });
        if (node.indices) for (const idx of node.indices) walk(idx);
        return;
      case 'BufferStoreNode':
        if (node.buffer) accesses.push({ name: node.buffer.name, kind: 'write', ranges: indexRanges(node.indices, loops) });
        if (node.indices) for (const idx of node.indices) walk(idx);
        walk(node.value);
        return;
      case 'MathOpNode': walk(node.a); if (node.b) walk(node.b); return;
      case 'CompareNode': walk(node.a); walk(node.b); return;
      case 'CallExternNode': for (const a of node.args) walk(a); return;
      case 'IfThenElseNode': walk(node.condition); walk(node.thenBody); if (node.elseBody) walk(node.elseBody); return;
      case 'CastNode': walk(node.expr); return;
      case 'LetStmtNode': walk(node.value); walk(node.body); return;
      case 'SeqNode': for (const s of node.stmts) walk(s); return;
      case 'BlockNode': if (node.initBody) walk(node.initBody); walk(node.body); return;
      default: return;
    }
  };
  walk(blockNode.body);
  if (blockNode.initBody) walk(blockNode.initBody);
  return accesses;
}

export function rangesOverlap(ra, rb) {
  if (!ra || !rb || ra.length !== rb.length) return true;
  for (let i = 0; i < ra.length; i++) {
    const a = ra[i], b = rb[i];
    if (a === null || b === null) continue;
    if (a[0] + a[1] <= b[0] || b[0] + b[1] <= a[0]) return false;
  }
  return true;
}

function accessesConflict(accA, accB, kindFilter) {
  for (const a of accA) {
    for (const b of accB) {
      if (a.name !== b.name) continue;
      if (a.kind === 'read' && b.kind === 'read') continue;
      if (kindFilter && !kindFilter(a, b)) continue;
      if (rangesOverlap(a.ranges, b.ranges)) return true;
    }
  }
  return false;
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
    const seen = new Set();
    for (const a of srcAcc.accesses) {
      for (const b of dstAcc.accesses) {
        if (a.name !== b.name) continue;
        if (!rangesOverlap(a.ranges, b.ranges)) continue;
        let kind = null;
        if (a.kind === 'write' && b.kind === 'read') kind = DepKind.RAW;
        else if (a.kind === 'write' && b.kind === 'write') kind = DepKind.WAW;
        else if (a.kind === 'read' && b.kind === 'write') kind = DepKind.WAR;
        if (!kind) continue;
        const key = `${kind}:${a.name}`;
        if (seen.has(key)) continue;
        seen.add(key);
        deps.push(new Dependency(srcBlock, dstBlock, kind, a.name));
      }
    }
    return deps;
  }

  canReorder(blockA, blockB) {
    const srcAcc = this._blockAccesses.get(blockA);
    const dstAcc = this._blockAccesses.get(blockB);
    if (!srcAcc || !dstAcc) return true;
    return !accessesConflict(srcAcc.accesses, dstAcc.accesses);
  }
}
