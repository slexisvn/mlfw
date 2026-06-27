import { walk as irWalk, collect as irCollect } from '../../ir/ir_visitor.js';

export class InplaceCandidate {
  constructor(srcBuffer, dstBuffer, reason) {
    this.srcBuffer = srcBuffer;
    this.dstBuffer = dstBuffer;
    this.reason = reason;
  }
}

export class InplaceAnalysis {
  static analyze(primFunc, livenessResult, allowedDonationParams = new Set()) {
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

    const alreadyAliased = new Set();

    for (const block of blocks) {
      const currentIdx = blockIdx.get(block);
      if (currentIdx === undefined) continue;

      for (const writeEntry of block.writes) {
        const dstBuf = writeEntry.buffer;
        if (livenessResult.isParam(dstBuf) && !allowedDonationParams.has(dstBuf)) continue;

        for (const readEntry of block.reads) {
          const srcBuf = readEntry.buffer;
          if (srcBuf === dstBuf) continue;
          if (alreadyAliased.has(srcBuf)) continue;
          if (livenessResult.isParam(srcBuf) && !allowedDonationParams.has(srcBuf)) continue;

          if (!shapesMatch(srcBuf, dstBuf)) continue;
          if (srcBuf.dtype !== dstBuf.dtype) continue;
          if (srcBuf.scope !== dstBuf.scope) continue;

          const srcInterval = livenessResult.intervals.get(srcBuf);
          const dstInterval = livenessResult.intervals.get(dstBuf);
          if (!srcInterval || !dstInterval) continue;

          if (srcInterval.lastUse <= dstInterval.firstUse) {
            const lastRead = bufferLastReadIdx.get(srcBuf);
            if (lastRead === undefined || lastRead <= currentIdx) {
              if (!isInplaceComputeSafe(block, srcBuf, dstBuf)) continue;
              candidates.push(new InplaceCandidate(
                srcBuf, dstBuf,
                `${srcBuf.name} last used at ${srcInterval.lastUse}, ${dstBuf.name} first used at ${dstInterval.firstUse}`
              ));
              alreadyAliased.add(srcBuf);
              break;
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


function exprEqual(x, y) {
  if (x === y) return true;
  if (!x || !y || typeof x !== 'object' || typeof y !== 'object') return false;
  if (x.type !== y.type) return false;
  switch (x.type) {
    case 'VariableNode': return x.name === y.name;
    case 'IntImmNode': return x.value === y.value;
    case 'FloatImmNode': return x.value === y.value;
    case 'MathOpNode': return x.op === y.op && exprEqual(x.a, y.a) && exprEqual(x.b, y.b);
    case 'CompareNode': return x.direction === y.direction && exprEqual(x.a, y.a) && exprEqual(x.b, y.b);
    case 'CastNode': return x.toDtype === y.toDtype && exprEqual(x.expr, y.expr);
    case 'BufferLoadNode': return x.buffer === y.buffer && indexListEqual(x.indices, y.indices);
    default: return false;
  }
}

function indexListEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!exprEqual(a[i], b[i])) return false;
  }
  return true;
}

function walkNodes(root, visit) {
  if (root) irWalk(root, visit);
}

function isInplaceComputeSafe(block, srcBuf, dstBuf) {
  const dstStores = [];
  const srcLoads = [];
  for (const root of [block.body, block.initBody]) {
    if (!root) continue;
    walkNodes(root, (node) => {
      if (node.type === 'BufferStoreNode' && node.buffer === dstBuf) dstStores.push(node);
      else if (node.type === 'BufferLoadNode' && node.buffer === srcBuf) srcLoads.push(node);
    });
  }

  if (dstStores.length !== 1) return false;
  if (srcLoads.length === 0) return false;

  const store = dstStores[0];
  const writeIndex = store.indices;
  for (const load of srcLoads) {
    if (!indexListEqual(load.indices, writeIndex)) return false;
  }

  const loadsInsideStore = new Set();
  for (const root of [store.value, ...store.indices]) {
    if (!root) continue;
    walkNodes(root, (node) => {
      if (node.type === 'BufferLoadNode' && node.buffer === srcBuf) loadsInsideStore.add(node);
    });
  }

  let allInside = true;
  for (const load of srcLoads) {
    if (!loadsInsideStore.has(load)) { allInside = false; break; }
  }
  if (!allInside) {
    const v = store.value;
    const pureCopy = v && v.type === 'BufferLoadNode' && v.buffer === srcBuf && indexListEqual(v.indices, writeIndex);
    if (!pureCopy) return false;
  }

  return true;
}

function collectBlocks(node, result) {
  if (!node) return;
  for (const b of irCollect(node, (n) => n.type === 'BlockNode', { kinds: 'stmt' })) result.push(b);
}
