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
    };

    const visit = (node) => {
      if (!node) return;

      switch (node.type) {
        case 'SeqNode':
          for (const s of node.stmts) visit(s);
          break;

        case 'ForNode':
          visit(node.body);
          break;

        case 'BlockNode':
          stmtOrder.push({ idx: stmtIdx, node });
          for (const r of node.reads) touch(r.buffer);
          for (const w of node.writes) touch(w.buffer);
          visitBody(node.body);
          if (node.initBody) visitBody(node.initBody);
          stmtIdx++;
          break;

        case 'AllocateNode':
          touch(node.buffer);
          visit(node.body);
          break;

        case 'IfThenElseNode':
          visit(node.thenBody);
          if (node.elseBody) visit(node.elseBody);
          break;

        case 'LetStmtNode':
          visit(node.body);
          break;

        default:
          visitBody(node);
          break;
      }
    };

    const visitBody = (node) => {
      if (!node) return;
      if (node.type === 'BufferStoreNode') {
        touch(node.buffer);
        visitExpr(node.value);
        for (const idx of node.indices) visitExpr(idx);
      } else if (node.type === 'BufferLoadNode') {
        touch(node.buffer);
        for (const idx of node.indices) visitExpr(idx);
      } else if (node.type === 'SeqNode') {
        for (const s of node.stmts) visitBody(s);
      }
    };

    const visitExpr = (node) => {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'BufferLoadNode') {
        touch(node.buffer);
        return;
      }
      if (node.type === 'CallExternNode') {
        for (const a of node.args) visitExpr(a);
        return;
      }
      if (node.a) visitExpr(node.a);
      if (node.b) visitExpr(node.b);
    };

    visit(primFunc.body);

    return new BufferLivenessResult(intervals, stmtOrder, paramBuffers);
  }
}
