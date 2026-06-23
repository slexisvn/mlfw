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

    const touchExpr = (node) => {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'BufferLoadNode') { touch(node.buffer); return; }
      if (node.type === 'CallExternNode') { for (const a of node.args) touchExpr(a); return; }
      if (node.a) touchExpr(node.a);
      if (node.b) touchExpr(node.b);
      if (node.expr) touchExpr(node.expr);
      if (node.cond) touchExpr(node.cond);
      if (node.condition) touchExpr(node.condition);
      if (node.thenBody) touchExpr(node.thenBody);
      if (node.elseBody) touchExpr(node.elseBody);
    };

    const touchBody = (node) => {
      if (!node) return;
      if (node.type === 'BufferStoreNode') {
        touch(node.buffer);
        touchExpr(node.value);
        for (const idx of node.indices) touchExpr(idx);
      } else if (node.type === 'BufferLoadNode') {
        touch(node.buffer);
        for (const idx of node.indices) touchExpr(idx);
      } else if (node.type === 'SeqNode') {
        for (const s of node.stmts) touchBody(s);
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
          walk(node.body);
          const bodyEnd = stmtIdx > bodyStart ? stmtIdx - 1 : bodyStart;
          extendRegion(logStart, bodyEnd);
          break;
        }
        case 'BlockNode':
          stmtOrder.push({ idx: stmtIdx, node });
          for (const r of node.reads) touch(r.buffer);
          for (const w of node.writes) touch(w.buffer);
          touchBody(node.body);
          if (node.initBody) touchBody(node.initBody);
          stmtIdx++;
          break;
        case 'AllocateNode':
          touch(node.buffer);
          walk(node.body);
          break;
        case 'IfThenElseNode': {
          const branchStart = stmtIdx;
          const logStart = touchLog.length;
          walk(node.thenBody);
          if (node.elseBody) walk(node.elseBody);
          const branchEnd = stmtIdx > branchStart ? stmtIdx - 1 : branchStart;
          extendRegion(logStart, branchEnd);
          break;
        }
        case 'LetStmtNode':
          walk(node.body);
          break;
        default:
          touchBody(node);
          break;
      }
    };

    walk(primFunc.body);

    return new BufferLivenessResult(intervals, stmtOrder, paramBuffers);
  }
}
