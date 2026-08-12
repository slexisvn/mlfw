import type { Buffer } from '../../ir/tensor/buffer.js';
import type { BufferInterval } from './buffer_liveness.js';
import type { InplaceCandidate } from './inplace_analysis.js';

export type OffsetEntry = { offset: number; size: number };
export type PlacedEntry = OffsetEntry & { firstUse: number; lastUseEff: number };
export type BufferAssignmentEntry = {
  offset: number;
  size: number;
  scope: string;
  pool: string;
  inplaceOf: Buffer | null;
  isDynamic?: boolean;
};
export type AllocStrategy = 'best-fit' | 'first-fit';
type GapSelector = {
  best: { offset: number; gap: number } | null;
  consider(start: number, gap: number, size: number): number | null;
  result(fallbackStart: number): number;
};

function insertByOffset<T extends OffsetEntry>(arr: T[], entry: T): void {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].offset <= entry.offset) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, entry);
}

function gapSelector(strategy: string): GapSelector {
  return {
    best: null,
    consider(start: number, gap: number, size: number): number | null {
      if (gap < size) return null;
      if (strategy === 'best-fit') {
        if (this.best === null || gap < this.best.gap) this.best = { offset: start, gap };
        return null;
      }
      return start;
    },
    result(fallbackStart: number): number {
      return this.best !== null ? this.best.offset : fallbackStart;
    },
  };
}

export class MemoryBlock {
  offset: number;
  size: number;
  buffer: Buffer;

  constructor(offset: number, size: number, buffer: Buffer) {
    this.offset = offset;
    this.size = size;
    this.buffer = buffer;
  }

  get end(): number {
    return this.offset + this.size;
  }

  overlaps(other: MemoryBlock): boolean {
    return this.offset < other.end && other.offset < this.end;
  }
}

export class MemoryPool {
  scope: string;
  alignment: number;
  strategy: string;
  blocks: MemoryBlock[];
  peakUsage: number;

  constructor(scope: string, alignment = 64, strategy = 'best-fit') {
    this.scope = scope;
    this.alignment = alignment;
    this.strategy = strategy;
    this.blocks = [];
    this.peakUsage = 0;
  }

  allocate(size: number, buffer: Buffer): MemoryBlock {
    const aligned = this._align(size);
    const offset = this._findFreeOffset(aligned);
    return this.placeAt(offset, aligned, buffer);
  }

  placeAt(offset: number, size: number, buffer: Buffer): MemoryBlock {
    const aligned = this._align(size);
    const block = new MemoryBlock(offset, aligned, buffer);
    insertByOffset(this.blocks, block);
    const end = offset + aligned;
    if (end > this.peakUsage) this.peakUsage = end;
    return block;
  }

  _align(size: number): number {
    return Math.ceil(size / this.alignment) * this.alignment;
  }

  _findFreeOffset(size: number): number {
    const live = this.blocks;
    let cursor = 0;
    const sel = gapSelector(this.strategy);

    for (const block of live) {
      const start = this._align(cursor);
      const gap = block.offset - start;
      const hit = sel.consider(start, gap, size);
      if (hit !== null) return hit;
      if (block.end > cursor) cursor = block.end;
    }

    return sel.result(this._align(cursor));
  }

  fragmentation(): number {
    if (this.peakUsage === 0) return 0;
    const used = this.blocks.reduce((sum: number, b: MemoryBlock) => sum + b.size, 0);
    return Math.max(0, 1 - used / this.peakUsage);
  }

  release(block: MemoryBlock): void {
    const idx = this.blocks.indexOf(block);
    if (idx >= 0) this.blocks.splice(idx, 1);
  }
}

export class BufferAssignment {
  assignments: Map<Buffer, BufferAssignmentEntry>;
  pools: Map<string, MemoryPool>;
  inplaceMap: Map<Buffer, Buffer>;
  effLastUse: Map<Buffer, number>;

  constructor() {
    this.assignments = new Map();
    this.pools = new Map();
    this.inplaceMap = new Map();
    this.effLastUse = new Map();
  }

  assign(intervals: readonly BufferInterval[], inplaceCandidates: readonly InplaceCandidate[] = [], alignment = 64, strategy = 'best-fit'): this {
    for (const candidate of inplaceCandidates) {
      this.inplaceMap.set(candidate.dstBuffer, candidate.srcBuffer);
    }

    const ivByBuf = new Map<Buffer, BufferInterval>();
    for (const iv of intervals) ivByBuf.set(iv.buffer, iv);
    const effLastUse = new Map<Buffer, number>();
    for (const iv of intervals) effLastUse.set(iv.buffer, iv.lastUse);
    const children = new Map<Buffer, Buffer[]>();
    for (const [dst, src] of this.inplaceMap) {
      if (!ivByBuf.has(dst) || !ivByBuf.has(src)) continue;
      if (!children.has(src)) children.set(src, []);
      (children.get(src) as Buffer[]).push(dst);
    }
    if (children.size > 0) {
      const state = new Map<Buffer, number>();
      for (const iv of intervals) {
        const root = iv.buffer;
        if (state.get(root) === 1) continue;
        const stack: Buffer[] = [root];
        while (stack.length) {
          const node = stack[stack.length - 1];
          const kids = children.get(node);
          if (state.get(node) === undefined) {
            state.set(node, 0);
            if (kids) for (const c of kids) if (state.get(c) === undefined) stack.push(c);
            continue;
          }
          stack.pop();
          if (state.get(node) === 1) continue;
          if (kids) {
            let m = effLastUse.get(node) as number;
            for (const c of kids) { const cv = effLastUse.get(c) as number; if (cv > m) m = cv; }
            effLastUse.set(node, m);
          }
          state.set(node, 1);
        }
      }
    }
    this.effLastUse = effLastUse;

    const sorted = [...intervals].sort((a, b) => {
      const aSize = a.size;
      const bSize = b.size;
      const aStatic = aSize > 0;
      const bStatic = bSize > 0;
      if (aStatic && bStatic) {
        const sizeDiff = bSize - aSize;
        if (sizeDiff !== 0) return sizeDiff;
      } else if (aStatic !== bStatic) {
        return aStatic ? -1 : 1;
      }
      return a.firstUse - b.firstUse;
    });

    const activeByScope = new Map<string, PlacedEntry[]>();

    for (const interval of sorted) {
      const buf = interval.buffer;

      if (this.inplaceMap.has(buf)) {
        const srcBuf = this.inplaceMap.get(buf) as Buffer;
        const srcAssignment = this.assignments.get(srcBuf);
        if (srcAssignment) {
          this.assignments.set(buf, {
            offset: srcAssignment.offset,
            size: interval.size,
            scope: interval.scope,
            pool: srcAssignment.pool,
            inplaceOf: srcBuf
          });
          continue;
        }
      }

      const size = interval.size;
      if (size === 0) continue;
      if (size < 0) {
        this.assignments.set(buf, {
          offset: 0,
          size: 0,
          scope: interval.scope,
          pool: interval.scope,
          inplaceOf: null,
          isDynamic: true
        });
        continue;
      }

      const scope = interval.scope;
      let pool = this.pools.get(scope);
      if (!pool) {
        pool = new MemoryPool(scope, alignment);
        this.pools.set(scope, pool);
      }

      if (!activeByScope.has(scope)) activeByScope.set(scope, []);
      const placed = activeByScope.get(scope) as PlacedEntry[];
      const lastUseEff = effLastUse.get(buf) ?? interval.lastUse;
      const offset = this._interferenceOffset(placed, interval.firstUse, lastUseEff, pool._align(size), alignment, strategy);
      const block = pool.placeAt(offset, size, buf);
      insertByOffset(placed, { firstUse: interval.firstUse, lastUseEff, offset: block.offset, size: block.size });

      this.assignments.set(buf, {
        offset: block.offset,
        size: block.size,
        scope,
        pool: scope,
        inplaceOf: null
      });
    }

    return this;
  }

  _interferenceOffset(placed: readonly PlacedEntry[], firstUse: number, lastUseEff: number, size: number, alignment: number, strategy: string): number {
    const ranges: [number, number][] = [];
    for (const p of placed) {
      if (p.firstUse <= lastUseEff && firstUse <= p.lastUseEff) {
        ranges.push([p.offset, p.offset + p.size]);
      }
    }

    let cursor = 0;
    const sel = gapSelector(strategy);
    for (const [lo, hi] of ranges) {
      const start = Math.ceil(cursor / alignment) * alignment;
      const gap = lo - start;
      const hit = sel.consider(start, gap, size);
      if (hit !== null) return hit;
      if (hi > cursor) cursor = hi;
    }
    return sel.result(Math.ceil(cursor / alignment) * alignment);
  }

  getOffset(buffer: Buffer): number {
    const entry = this.assignments.get(buffer);
    return entry ? entry.offset : -1;
  }

  getAssignment(buffer: Buffer): BufferAssignmentEntry | null {
    return this.assignments.get(buffer) || null;
  }

  peakMemory(scope: string | null = null): number {
    if (scope) {
      const pool = this.pools.get(scope);
      return pool ? pool.peakUsage : 0;
    }
    let total = 0;
    for (const [, pool] of this.pools) {
      total += pool.peakUsage;
    }
    return total;
  }
}
