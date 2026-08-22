import { walk as irWalk, collect as irCollect } from '../../ir/ir_visitor.js';
import { toLinearForm } from '../../analysis/iter_map.js';
import type { Buffer } from '../../ir/tensor/buffer.js';
import type { Dim } from '../../ir/graph/types.js';
import type { BlockNode, BufferLoadNode, BufferStoreNode, CastNode, CompareNode, FloatImmNode, IntImmNode, MathOpNode, PrimFunc, TirNode, VariableNode } from '../../ir/tensor/nodes.js';
import type { IRNode } from '../../ir/ir_visitor.js';
import type { BufferLivenessResult } from './buffer_liveness.js';

export class InplaceCandidate {
  srcBuffer: Buffer;
  dstBuffer: Buffer;
  reason: string;

  constructor(srcBuffer: Buffer, dstBuffer: Buffer, reason: string) {
    this.srcBuffer = srcBuffer;
    this.dstBuffer = dstBuffer;
    this.reason = reason;
  }
}

export class InplaceAnalysis {
  static analyze(primFunc: PrimFunc, livenessResult: BufferLivenessResult, allowedDonationParams: ReadonlySet<Buffer> = new Set()): InplaceCandidate[] {
    const candidates: InplaceCandidate[] = [];

    const blocks: BlockNode[] = [];
    collectBlocks(primFunc.body, blocks);

    const blockIdx = new Map<BlockNode, number>();
    for (const entry of livenessResult.stmtOrder) {
      blockIdx.set(entry.node, entry.idx);
    }

    const bufferLastReadIdx = new Map<Buffer, number>();
    for (const entry of livenessResult.stmtOrder) {
      for (const r of entry.node.reads) {
        const prev = bufferLastReadIdx.get(r.buffer);
        if (prev === undefined || entry.idx > prev) {
          bufferLastReadIdx.set(r.buffer, entry.idx);
        }
      }
    }

    const alreadyAliased = new Set<Buffer>();

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

          if (!layoutsMatch(srcBuf, dstBuf)) continue;
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

function dimsMatch(a: readonly Dim[], b: readonly Dim[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function layoutsMatch(a: Buffer, b: Buffer): boolean {
  return dimsMatch(a.shape, b.shape) && dimsMatch(a.strides, b.strides);
}


function affineEqual(x: TirNode, y: TirNode): boolean | null {
  const fx = toLinearForm(x);
  if (!fx) return null;
  const fy = toLinearForm(y);
  if (!fy) return null;
  if (fx.offset !== fy.offset || fx.terms.size !== fy.terms.size) return false;
  for (const [name, coeff] of fx.terms) {
    if (fy.terms.get(name) !== coeff) return false;
  }
  return true;
}

function exprEqual(x: TirNode, y: TirNode): boolean {
  if (x === y) return true;
  if (!x || !y || typeof x !== 'object' || typeof y !== 'object') return false;
  const affine = affineEqual(x, y);
  if (affine !== null) return affine;
  if (x.type !== y.type) return false;
  switch (x.type) {
    case 'VariableNode': return (x as VariableNode).name === (y as VariableNode).name;
    case 'IntImmNode': return (x as IntImmNode).value === (y as IntImmNode).value;
    case 'FloatImmNode': return (x as FloatImmNode).value === (y as FloatImmNode).value;
    case 'MathOpNode': {
      const a = x as MathOpNode, b = y as MathOpNode;
      return a.op === b.op && exprEqual(a.a, b.a) && exprEqual(a.b as TirNode, b.b as TirNode);
    }
    case 'CompareNode': {
      const a = x as CompareNode, b = y as CompareNode;
      return a.direction === b.direction && exprEqual(a.a, b.a) && exprEqual(a.b, b.b);
    }
    case 'CastNode': {
      const a = x as CastNode, b = y as CastNode;
      return a.toDtype === b.toDtype && exprEqual(a.expr, b.expr);
    }
    case 'BufferLoadNode': {
      const a = x as BufferLoadNode, b = y as BufferLoadNode;
      return a.buffer === b.buffer && indexListEqual(a.indices, b.indices);
    }
    default: return false;
  }
}

function indexListEqual(a: readonly TirNode[], b: readonly TirNode[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!exprEqual(a[i], b[i])) return false;
  }
  return true;
}

function walkNodes(root: TirNode | null | undefined, visit: (node: TirNode) => void): void {
  if (root) irWalk(root as unknown as IRNode, visit as never);
}

function isInplaceComputeSafe(block: BlockNode, srcBuf: Buffer, dstBuf: Buffer): boolean {
  const dstStores: BufferStoreNode[] = [];
  const srcLoads: BufferLoadNode[] = [];
  for (const root of [block.body, block.initBody]) {
    if (!root) continue;
    walkNodes(root, (node: TirNode) => {
      if (node.type === 'BufferStoreNode' && (node as BufferStoreNode).buffer === dstBuf) dstStores.push(node as BufferStoreNode);
      else if (node.type === 'BufferLoadNode' && (node as BufferLoadNode).buffer === srcBuf) srcLoads.push(node as BufferLoadNode);
    });
  }

  if (dstStores.length !== 1) return false;
  if (srcLoads.length === 0) return false;

  const store = dstStores[0];
  const writeIndex = store.indices;
  for (const load of srcLoads) {
    if (!indexListEqual(load.indices, writeIndex)) return false;
  }

  const loadsInsideStore = new Set<BufferLoadNode>();
  for (const root of [store.value, ...store.indices]) {
    if (!root) continue;
    walkNodes(root, (node: TirNode) => {
      if (node.type === 'BufferLoadNode' && (node as BufferLoadNode).buffer === srcBuf) loadsInsideStore.add(node as BufferLoadNode);
    });
  }

  let allInside = true;
  for (const load of srcLoads) {
    if (!loadsInsideStore.has(load)) { allInside = false; break; }
  }
  if (!allInside) {
    const v = store.value as BufferLoadNode;
    const pureCopy = v && v.type === 'BufferLoadNode' && v.buffer === srcBuf && indexListEqual(v.indices, writeIndex);
    if (!pureCopy) return false;
  }

  return true;
}

function collectBlocks(node: TirNode | null | undefined, result: BlockNode[]): void {
  if (!node) return;
  for (const b of irCollect(node as unknown as IRNode, (n: IRNode) => n.type === 'BlockNode', { kinds: 'stmt' })) result.push(b as unknown as BlockNode);
}
