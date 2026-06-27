function insertByOffset(arr, entry) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].offset <= entry.offset) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, entry);
}

function gapSelector(strategy) {
  return {
    best: null,
    consider(start, gap, size) {
      if (gap < size) return null;
      if (strategy === 'best-fit') {
        if (this.best === null || gap < this.best.gap) this.best = { offset: start, gap };
        return null;
      }
      return start;
    },
    result(fallbackStart) {
      return this.best !== null ? this.best.offset : fallbackStart;
    },
  };
}

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
    const offset = this._findFreeOffset(aligned);
    return this.placeAt(offset, aligned, buffer);
  }

  placeAt(offset, size, buffer) {
    const aligned = this._align(size);
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

  assign(intervals, inplaceCandidates = [], alignment = 64, strategy = 'best-fit') {
    for (const candidate of inplaceCandidates) {
      this.inplaceMap.set(candidate.dstBuffer, candidate.srcBuffer);
    }

    const ivByBuf = new Map();
    for (const iv of intervals) ivByBuf.set(iv.buffer, iv);
    const effLastUse = new Map();
    for (const iv of intervals) effLastUse.set(iv.buffer, iv.lastUse);
    const children = new Map();
    for (const [dst, src] of this.inplaceMap) {
      if (!ivByBuf.has(dst) || !ivByBuf.has(src)) continue;
      if (!children.has(src)) children.set(src, []);
      children.get(src).push(dst);
    }
    if (children.size > 0) {
      const state = new Map();
      for (const iv of intervals) {
        const root = iv.buffer;
        if (state.get(root) === 1) continue;
        const stack = [root];
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
            let m = effLastUse.get(node);
            for (const c of kids) { const cv = effLastUse.get(c); if (cv > m) m = cv; }
            effLastUse.set(node, m);
          }
          state.set(node, 1);
        }
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
      const placed = activeByScope.get(scope);
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

  _interferenceOffset(placed, firstUse, lastUseEff, size, alignment, strategy) {
    const ranges = [];
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
