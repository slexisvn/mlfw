export class MemoryBlock {
  constructor(offset, size, buffer) {
    this.offset = offset;
    this.size = size;
    this.buffer = buffer;
  }

  get end() {
    return this.offset + this.size;
  }

  overlaps(other) {
    return this.offset < other.end && other.offset < this.end;
  }
}

export class MemoryPool {
  constructor(scope, alignment = 64, strategy = 'best-fit') {
    this.scope = scope;
    this.alignment = alignment;
    this.strategy = strategy;
    this.blocks = [];
    this.peakUsage = 0;
  }

  allocate(size, buffer) {
    const aligned = this._align(size);
    const offset = this._findFreeOffset(aligned, buffer);
    const block = new MemoryBlock(offset, aligned, buffer);
    this.blocks.push(block);
    const end = offset + aligned;
    if (end > this.peakUsage) this.peakUsage = end;
    return block;
  }

  _align(size) {
    return Math.ceil(size / this.alignment) * this.alignment;
  }

  _findFreeOffset(size) {
    const live = this.blocks.slice().sort((a, b) => a.offset - b.offset);
    let cursor = 0;
    let best = null;

    for (const block of live) {
      const start = this._align(cursor);
      const gap = block.offset - start;
      if (gap >= size) {
        if (this.strategy === 'best-fit') {
          if (best === null || gap < best.gap) best = { offset: start, gap };
        } else {
          return start;
        }
      }
      if (block.end > cursor) cursor = block.end;
    }

    return best !== null ? best.offset : this._align(cursor);
  }

  fragmentation() {
    if (this.peakUsage === 0) return 0;
    const used = this.blocks.reduce((sum, b) => sum + b.size, 0);
    return Math.max(0, 1 - used / this.peakUsage);
  }

  release(block) {
    const idx = this.blocks.indexOf(block);
    if (idx >= 0) this.blocks.splice(idx, 1);
  }
}

export class BufferAssignment {
  constructor() {
    this.assignments = new Map();
    this.pools = new Map();
    this.inplaceMap = new Map();
  }

  assign(intervals, inplaceCandidates = [], alignment = 64) {
    for (const candidate of inplaceCandidates) {
      this.inplaceMap.set(candidate.dstBuffer, candidate.srcBuffer);
    }

    const ivByBuf = new Map();
    for (const iv of intervals) ivByBuf.set(iv.buffer, iv);
    const effLastUse = new Map();
    for (const iv of intervals) effLastUse.set(iv.buffer, iv.lastUse);
    for (const [dst] of this.inplaceMap) {
      const dstIv = ivByBuf.get(dst);
      if (!dstIv) continue;
      let cur = this.inplaceMap.get(dst);
      const seen = new Set();
      while (cur && ivByBuf.has(cur) && !seen.has(cur)) {
        seen.add(cur);
        if (dstIv.lastUse > effLastUse.get(cur)) effLastUse.set(cur, dstIv.lastUse);
        cur = this.inplaceMap.get(cur);
      }
    }

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

    const activeByScope = new Map();

    for (const interval of sorted) {
      const buf = interval.buffer;

      if (this.inplaceMap.has(buf)) {
        const srcBuf = this.inplaceMap.get(buf);
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
      const active = activeByScope.get(scope);

      for (let i = active.length - 1; i >= 0; i--) {
        const eff = effLastUse.get(active[i].interval.buffer) ?? active[i].interval.lastUse;
        if (eff < interval.firstUse) {
          pool.release(active[i].block);
          active.splice(i, 1);
        }
      }

      const block = pool.allocate(size, buf);
      active.push({ interval, block });

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

  getOffset(buffer) {
    const entry = this.assignments.get(buffer);
    return entry ? entry.offset : -1;
  }

  getAssignment(buffer) {
    return this.assignments.get(buffer) || null;
  }

  peakMemory(scope = null) {
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
