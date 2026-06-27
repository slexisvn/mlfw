export class BufferInterval {
  constructor(buffer, firstUse, lastUse, scope) {
    this.buffer = buffer;
    this.firstUse = firstUse;
    this.lastUse = lastUse;
    this.scope = scope;
  }

  get size() {
    return this.buffer.sizeInBytes();
  }

  overlaps(other) {
    return this.firstUse <= other.lastUse && other.firstUse <= this.lastUse;
  }
}

export class BufferLivenessResult {
  constructor(intervals, stmtOrder, paramBuffers) {
    this.intervals = intervals;
    this.stmtOrder = stmtOrder;
    this.paramBuffers = paramBuffers;
  }

  isParam(buffer) {
    return this.paramBuffers.has(buffer);
  }

  getTemporaries() {
    const result = [];
    for (const [buf, interval] of this.intervals) {
      if (!this.paramBuffers.has(buf)) result.push(interval);
    }
    return result;
  }

  interfere(a, b) {
    const ia = this.intervals.get(a);
    const ib = this.intervals.get(b);
    if (!ia || !ib) return false;
    return ia.overlaps(ib);
  }
}

const META_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);

function isBuffer(x) {
  return !!x && typeof x === 'object'
    && typeof x.name === 'string'
    && x.dtype !== undefined
    && x.shape !== undefined
    && x.type === undefined;
}

export class BufferLiveness {
  static analyze(primFunc) {
    const intervals = new Map();
    const stmtOrder = [];
    let stmtIdx = 0;

    const paramBuffers = new Set();
    for (const [, buf] of primFunc.bufferMap) {
      paramBuffers.add(buf);
    }

    const touchLog = [];

    const touch = (buffer) => {
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

    const touchAll = (node, seen) => {
      if (!node || typeof node !== 'object' || seen.has(node)) return;
      seen.add(node);
      if (isBuffer(node)) { touch(node); return; }
      if (isBuffer(node.buffer)) touch(node.buffer);
      for (const key of Object.keys(node)) {
        if (META_KEYS.has(key) || key === 'buffer') continue;
        const value = node[key];
        if (!value || typeof value !== 'object') continue;
        if (Array.isArray(value)) {
          for (const item of value) touchAll(item, seen);
        } else {
          touchAll(value, seen);
        }
      }
    };

    const extendRegion = (logStart, endIdx) => {
      for (let i = logStart; i < touchLog.length; i++) {
        const interval = intervals.get(touchLog[i]);
        if (interval && endIdx > interval.lastUse) interval.lastUse = endIdx;
      }
    };

    const walk = (node) => {
      if (!node) return;

      switch (node.type) {
        case 'SeqNode':
          for (const s of node.stmts) walk(s);
          break;
        case 'ForNode': {
          const bodyStart = stmtIdx;
          const logStart = touchLog.length;
          touchAll(node.min, new Set());
          touchAll(node.extent, new Set());
          walk(node.body);
          const bodyEnd = stmtIdx > bodyStart ? stmtIdx - 1 : bodyStart;
          extendRegion(logStart, bodyEnd);
          break;
        }
        case 'WhileNode': {
          const bodyStart = stmtIdx;
          const logStart = touchLog.length;
          touchAll(node.condVar, new Set());
          walk(node.condBody);
          walk(node.loopBody);
          const bodyEnd = stmtIdx > bodyStart ? stmtIdx - 1 : bodyStart;
          extendRegion(logStart, bodyEnd);
          break;
        }
        case 'BlockNode':
          stmtOrder.push({ idx: stmtIdx, node });
          for (const r of node.reads) touch(r.buffer);
          for (const w of node.writes) touch(w.buffer);
          touchAll(node.body, new Set());
          if (node.initBody) touchAll(node.initBody, new Set());
          stmtIdx++;
          break;
        case 'AllocateNode':
          touch(node.buffer);
          walk(node.body);
          break;
        case 'IfThenElseNode': {
          const branchStart = stmtIdx;
          const logStart = touchLog.length;
          touchAll(node.condition, new Set());
          walk(node.thenBody);
          if (node.elseBody) walk(node.elseBody);
          const branchEnd = stmtIdx > branchStart ? stmtIdx - 1 : branchStart;
          extendRegion(logStart, branchEnd);
          break;
        }
        case 'LetStmtNode':
          touchAll(node.value, new Set());
          walk(node.body);
          break;
        case 'EvaluateNode':
          touchAll(node.value, new Set());
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
