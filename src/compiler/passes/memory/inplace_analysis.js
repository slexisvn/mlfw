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

    const blockIdx = new Map();
    for (const entry of livenessResult.stmtOrder) {
      blockIdx.set(entry.node, entry.idx);
    }

    const bufferLastReadIdx = new Map();
    for (const entry of livenessResult.stmtOrder) {
      for (const r of entry.node.reads) {
        const prev = bufferLastReadIdx.get(r.buffer);
        if (prev === undefined || entry.idx > prev) {
          bufferLastReadIdx.set(r.buffer, entry.idx);
        }
      }
    }

    for (const block of blocks) {
      const currentIdx = blockIdx.get(block);
      if (currentIdx === undefined) continue;

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
            const lastRead = bufferLastReadIdx.get(srcBuf);
            if (lastRead === undefined || lastRead <= currentIdx) {
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

function collectBlocks(node, result) {
  const stack = [node];
  while (stack.length > 0) {
    const cur = stack.pop();
    if (!cur) continue;
    if (cur.type === 'BlockNode') result.push(cur);
    if (cur.body) stack.push(cur.body);
    if (cur.stmts) for (const s of cur.stmts) stack.push(s);
    if (cur.thenBody) stack.push(cur.thenBody);
    if (cur.elseBody) stack.push(cur.elseBody);
    if (cur.initBody) stack.push(cur.initBody);
  }
}
