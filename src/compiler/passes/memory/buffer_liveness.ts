import type { Buffer } from '../../ir/tensor/buffer.js';
import type { AllocateNode, BlockNode, EvaluateNode, ForNode, IfThenElseNode, LetStmtNode, PrimFunc, SeqNode, TirNode, WhileNode } from '../../ir/tensor/nodes.js';

export type StmtOrderEntry = { idx: number; node: BlockNode };

export class BufferInterval {
  buffer: Buffer;
  firstUse: number;
  lastUse: number;
  scope: string;

  constructor(buffer: Buffer, firstUse: number, lastUse: number, scope: string) {
    this.buffer = buffer;
    this.firstUse = firstUse;
    this.lastUse = lastUse;
    this.scope = scope;
  }

  get size(): number {
    return this.buffer.sizeInBytes() as number;
  }

  overlaps(other: BufferInterval): boolean {
    return this.firstUse <= other.lastUse && other.firstUse <= this.lastUse;
  }
}

export class BufferLivenessResult {
  intervals: Map<Buffer, BufferInterval>;
  stmtOrder: StmtOrderEntry[];
  paramBuffers: Set<Buffer>;

  constructor(intervals: Map<Buffer, BufferInterval>, stmtOrder: StmtOrderEntry[], paramBuffers: Set<Buffer>) {
    this.intervals = intervals;
    this.stmtOrder = stmtOrder;
    this.paramBuffers = paramBuffers;
  }

  isParam(buffer: Buffer): boolean {
    return this.paramBuffers.has(buffer);
  }

  getTemporaries(): BufferInterval[] {
    const result: BufferInterval[] = [];
    for (const [buf, interval] of this.intervals) {
      if (!this.paramBuffers.has(buf)) result.push(interval);
    }
    return result;
  }

  interfere(a: Buffer, b: Buffer): boolean {
    const ia = this.intervals.get(a);
    const ib = this.intervals.get(b);
    if (!ia || !ib) return false;
    return ia.overlaps(ib);
  }
}

const META_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);

function isBuffer(x: unknown): x is Buffer {
  const b = x as Buffer & { type?: unknown };
  return !!x && typeof x === 'object'
    && typeof b.name === 'string'
    && b.dtype !== undefined
    && b.shape !== undefined
    && b.type === undefined;
}

export class BufferLiveness {
  static analyze(primFunc: PrimFunc): BufferLivenessResult {
    const intervals = new Map<Buffer, BufferInterval>();
    const stmtOrder: StmtOrderEntry[] = [];
    let stmtIdx = 0;

    const paramBuffers = new Set<Buffer>();
    for (const [, buf] of primFunc.bufferMap) {
      paramBuffers.add(buf);
    }

    const touchLog: Buffer[] = [];

    const touch = (buffer: Buffer | null | undefined): void => {
      if (!buffer) return;
      let interval = intervals.get(buffer);
      if (!interval) {
        interval = new BufferInterval(buffer, stmtIdx, stmtIdx, buffer.scope);
        intervals.set(buffer, interval);
      } else {
        if (stmtIdx < interval.firstUse) interval.firstUse = stmtIdx;
        if (stmtIdx > interval.lastUse) interval.lastUse = stmtIdx;
      }
      touchLog.push(buffer);
    };

    const touchAll = (node: unknown, seen: Set<unknown>): void => {
      if (!node || typeof node !== 'object' || seen.has(node)) return;
      seen.add(node);
      if (isBuffer(node)) { touch(node); return; }
      const slots = node as Record<string, unknown>;
      if (isBuffer(slots.buffer)) touch(slots.buffer);
      for (const key of Object.keys(slots)) {
        if (META_KEYS.has(key) || key === 'buffer') continue;
        const value = slots[key];
        if (!value || typeof value !== 'object') continue;
        if (Array.isArray(value)) {
          for (const item of value) touchAll(item, seen);
        } else {
          touchAll(value, seen);
        }
      }
    };

    const extendRegion = (logStart: number, endIdx: number): void => {
      for (let i = logStart; i < touchLog.length; i++) {
        const interval = intervals.get(touchLog[i]);
        if (interval && endIdx > interval.lastUse) interval.lastUse = endIdx;
      }
    };

    const walk = (node: TirNode | null | undefined): void => {
      if (!node) return;

      switch (node.type) {
        case 'SeqNode':
          for (const s of (node as SeqNode).stmts) walk(s);
          break;
        case 'ForNode': {
          const f = node as ForNode;
          const bodyStart = stmtIdx;
          const logStart = touchLog.length;
          touchAll(f.min, new Set());
          touchAll(f.extent, new Set());
          walk(f.body);
          const bodyEnd = stmtIdx > bodyStart ? stmtIdx - 1 : bodyStart;
          extendRegion(logStart, bodyEnd);
          break;
        }
        case 'WhileNode': {
          const w = node as WhileNode;
          const bodyStart = stmtIdx;
          const logStart = touchLog.length;
          touchAll(w.condVar, new Set());
          walk(w.condBody);
          walk(w.loopBody);
          const bodyEnd = stmtIdx > bodyStart ? stmtIdx - 1 : bodyStart;
          extendRegion(logStart, bodyEnd);
          break;
        }
        case 'BlockNode': {
          const blk = node as BlockNode;
          stmtOrder.push({ idx: stmtIdx, node: blk });
          for (const r of blk.reads) touch(r.buffer);
          for (const w of blk.writes) touch(w.buffer);
          touchAll(blk.body, new Set());
          if (blk.initBody) touchAll(blk.initBody, new Set());
          stmtIdx++;
          break;
        }
        case 'AllocateNode': {
          const al = node as AllocateNode;
          touch(al.buffer);
          walk(al.body);
          break;
        }
        case 'IfThenElseNode': {
          const ite = node as IfThenElseNode;
          const branchStart = stmtIdx;
          const logStart = touchLog.length;
          touchAll(ite.condition, new Set());
          walk(ite.thenBody);
          if (ite.elseBody) walk(ite.elseBody);
          const branchEnd = stmtIdx > branchStart ? stmtIdx - 1 : branchStart;
          extendRegion(logStart, branchEnd);
          break;
        }
        case 'LetStmtNode': {
          const let_ = node as LetStmtNode;
          touchAll(let_.value, new Set());
          walk(let_.body);
          break;
        }
        case 'EvaluateNode':
          touchAll((node as EvaluateNode).value, new Set());
          break;
        default:
          touchAll(node, new Set());
          break;
      }
    };

    walk(primFunc.body);

    return new BufferLivenessResult(intervals, stmtOrder, paramBuffers);
  }
}
