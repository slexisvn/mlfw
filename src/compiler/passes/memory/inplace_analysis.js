export class InplaceCandidate {
  constructor(srcBuffer, dstBuffer, reason) {
    this.srcBuffer = srcBuffer;
    this.dstBuffer = dstBuffer;
    this.reason = reason;
  }
}

export class InplaceAnalysis {
  static analyze(primFunc, livenessResult) {
    const candidates = [];

    const blocks = [];
    collectBlocks(primFunc.body, blocks);

    for (const block of blocks) {
      for (const writeEntry of block.writes) {
        const dstBuf = writeEntry.buffer;
        if (livenessResult.isParam(dstBuf)) continue;

        for (const readEntry of block.reads) {
          const srcBuf = readEntry.buffer;
          if (srcBuf === dstBuf) continue;
          if (livenessResult.isParam(srcBuf)) continue;

          if (!shapesMatch(srcBuf, dstBuf)) continue;
          if (srcBuf.dtype !== dstBuf.dtype) continue;
          if (srcBuf.scope !== dstBuf.scope) continue;

          const srcInterval = livenessResult.intervals.get(srcBuf);
          const dstInterval = livenessResult.intervals.get(dstBuf);
          if (!srcInterval || !dstInterval) continue;

          if (srcInterval.lastUse <= dstInterval.firstUse) {
            if (!hasOtherUsersAfter(srcBuf, block, livenessResult)) {
              candidates.push(new InplaceCandidate(
                srcBuf, dstBuf,
                `${srcBuf.name} last used at ${srcInterval.lastUse}, ${dstBuf.name} first used at ${dstInterval.firstUse}`
              ));
            }
          }
        }
      }
    }

    return candidates;
  }
}

function shapesMatch(a, b) {
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  return true;
}

function hasOtherUsersAfter(buffer, currentBlock, livenessResult) {
  const interval = livenessResult.intervals.get(buffer);
  if (!interval) return false;
  const currentIdx = findBlockIdx(currentBlock, livenessResult.stmtOrder);
  for (const entry of livenessResult.stmtOrder) {
    if (entry.idx <= currentIdx) continue;
    const block = entry.node;
    for (const r of block.reads) {
      if (r.buffer === buffer) return true;
    }
  }
  return false;
}

function findBlockIdx(block, stmtOrder) {
  for (const entry of stmtOrder) {
    if (entry.node === block) return entry.idx;
  }
  return -1;
}

function collectBlocks(node, result) {
  if (!node) return;
  if (node.type === 'BlockNode') {
    result.push(node);
  }
  if (node.body) collectBlocks(node.body, result);
  if (node.stmts) {
    for (const s of node.stmts) collectBlocks(s, result);
  }
  if (node.thenBody) collectBlocks(node.thenBody, result);
  if (node.elseBody) collectBlocks(node.elseBody, result);
  if (node.initBody) collectBlocks(node.initBody, result);
}
