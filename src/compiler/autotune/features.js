import { ForKind } from '../ir/tensor/nodes.js';

export const STATEMENT_FEATURE_SCHEMA = [
  'iterCount', 'depth',
  'parallelLoops', 'vectorizedLoops', 'unrolledLoops', 'threadBoundLoops', 'serialLoops',
  'threadBlockSize', 'gridSize', 'underReduction',
  'numMathOps', 'numExternCalls', 'numReads', 'numWrites',
  'stride1Accesses', 'stridedAccesses', 'reuseCount', 'touchedBytes'
];

class ScheduleFeatures {
  constructor(raw) {
    this.numLoops = raw.numLoops || 0;
    this.numBlocks = raw.numBlocks || 0;
    this.totalIterations = raw.totalIterations || 0;
    this.maxLoopDepth = raw.maxLoopDepth || 0;
    this.numParallelLoops = raw.numParallelLoops || 0;
    this.numVectorizedLoops = raw.numVectorizedLoops || 0;
    this.numUnrolledLoops = raw.numUnrolledLoops || 0;
    this.numThreadBound = raw.numThreadBound || 0;
    this.numSerialLoops = raw.numSerialLoops || 0;
    this.totalBufferBytes = raw.totalBufferBytes || 0;
    this.numBufferReads = raw.numBufferReads || 0;
    this.numBufferWrites = raw.numBufferWrites || 0;
    this.numMathOps = raw.numMathOps || 0;
    this.numExternCalls = raw.numExternCalls || 0;
    this.arithmeticIntensity = raw.arithmeticIntensity || 0;
    this.innermostExtent = raw.innermostExtent || 0;
    this.outermostExtent = raw.outermostExtent || 0;
    this.hasReduction = raw.hasReduction || false;
    this.reductionDepth = raw.reductionDepth || 0;
    this.threadBlockSize = raw.threadBlockSize || 0;
    this.gridSize = raw.gridSize || 0;
    this.strideOneAccesses = raw.strideOneAccesses || 0;
    this.nonStrideOneAccesses = raw.nonStrideOneAccesses || 0;
  }
}

export class FeatureExtractor {
  static extract(primFunc) {
    const ctx = {
      numLoops: 0, numBlocks: 0, totalIterations: 1,
      maxLoopDepth: 0, currentDepth: 0,
      numParallelLoops: 0, numVectorizedLoops: 0,
      numUnrolledLoops: 0, numThreadBound: 0, numSerialLoops: 0,
      totalBufferBytes: 0, buffersSeen: new Set(),
      numBufferReads: 0, numBufferWrites: 0,
      numMathOps: 0, numExternCalls: 0,
      hasReduction: false, reductionDepth: 0,
      threadBlockSize: 1, gridSize: 1,
      innermostExtent: 0, outermostExtent: 0,
      loopExtents: [],
      strideOneAccesses: 0, nonStrideOneAccesses: 0
    };

    FeatureExtractor._visitIterative(primFunc.body, ctx);

    const bytes = ctx.totalBufferBytes;
    const ops = ctx.numMathOps + ctx.numExternCalls;
    ctx.arithmeticIntensity = bytes > 0 ? ops / bytes : 0;

    return new ScheduleFeatures(ctx);
  }

  static extractStatements(primFunc) {
    const out = [];
    const loopStack = [];
    let reduction = 0;
    const stack = [{ node: primFunc.body, action: 'enter' }];
    while (stack.length > 0) {
      const { node, action } = stack.pop();
      if (!node) continue;
      if (action === 'leaveFor') { loopStack.pop(); continue; }
      if (action === 'leaveBlock') { reduction--; continue; }
      switch (node.type) {
        case 'ForNode':
          loopStack.push(node);
          stack.push({ node: null, action: 'leaveFor' });
          stack.push({ node: node.body, action: 'enter' });
          break;
        case 'BlockNode':
          if (node.initBody) {
            reduction++;
            stack.push({ node: null, action: 'leaveBlock' });
            stack.push({ node: node.initBody, action: 'enter' });
          }
          stack.push({ node: node.body, action: 'enter' });
          break;
        case 'SeqNode':
          for (let i = node.stmts.length - 1; i >= 0; i--) stack.push({ node: node.stmts[i], action: 'enter' });
          break;
        case 'AllocateNode':
        case 'LetStmtNode':
          stack.push({ node: node.body, action: 'enter' });
          break;
        case 'IfThenElseNode':
          if (node.elseBody) stack.push({ node: node.elseBody, action: 'enter' });
          stack.push({ node: node.thenBody, action: 'enter' });
          break;
        case 'BufferStoreNode':
          out.push(FeatureExtractor._statementVector(node, loopStack, reduction));
          break;
      }
    }
    return out;
  }

  static _statementVector(store, loopStack, reduction) {
    let iterCount = 1, par = 0, vec = 0, unr = 0, thr = 0, ser = 0, tbs = 1, grid = 1;
    for (const f of loopStack) {
      const ext = f.extent && f.extent.type === 'IntImmNode' ? f.extent.value : 1;
      iterCount *= ext;
      switch (f.kind) {
        case ForKind.PARALLEL: par++; break;
        case ForKind.VECTORIZED: vec++; break;
        case ForKind.UNROLLED: unr++; break;
        case ForKind.THREAD_BINDING:
          thr++;
          if (f.threadTag && f.threadTag.startsWith('threadIdx')) tbs *= ext;
          else if (f.threadTag && f.threadTag.startsWith('blockIdx')) grid *= ext;
          break;
        default: ser++; break;
      }
    }
    const arith = { math: 0, extern: 0 };
    FeatureExtractor._countExpr(store.value, arith);
    const loopVarNames = loopStack.map(f => f.loopVar.name);
    const accesses = [{ buffer: store.buffer, indices: store.indices }];
    FeatureExtractor._collectLoads(store.value, accesses);
    let stride1 = 0, strided = 0, reuse = 0, touched = 0;
    for (const acc of accesses) {
      touched += acc.buffer && acc.buffer.sizeInBytes ? acc.buffer.sizeInBytes() : 0;
      const last = acc.indices && acc.indices.length > 0 ? acc.indices[acc.indices.length - 1] : null;
      if (last && last.type === 'VariableNode') stride1++;
      else strided++;
      const used = new Set();
      if (acc.indices) for (const idx of acc.indices) FeatureExtractor._collectVars(idx, used);
      for (const name of loopVarNames) if (!used.has(name)) reuse++;
    }
    const fields = {
      iterCount, depth: loopStack.length,
      parallelLoops: par, vectorizedLoops: vec, unrolledLoops: unr,
      threadBoundLoops: thr, serialLoops: ser,
      threadBlockSize: tbs, gridSize: grid,
      underReduction: reduction > 0 ? 1 : 0,
      numMathOps: arith.math, numExternCalls: arith.extern,
      numReads: accesses.length - 1, numWrites: 1,
      stride1Accesses: stride1, stridedAccesses: strided,
      reuseCount: reuse, touchedBytes: touched
    };
    return STATEMENT_FEATURE_SCHEMA.map(n => fields[n] || 0);
  }

  static _countExpr(node, acc) {
    if (!node || typeof node !== 'object') return;
    switch (node.type) {
      case 'MathOpNode':
        acc.math++;
        FeatureExtractor._countExpr(node.a, acc);
        if (node.b) FeatureExtractor._countExpr(node.b, acc);
        break;
      case 'CompareNode':
        acc.math++;
        FeatureExtractor._countExpr(node.a, acc);
        FeatureExtractor._countExpr(node.b, acc);
        break;
      case 'CallExternNode':
        acc.extern++;
        for (const a of node.args) FeatureExtractor._countExpr(a, acc);
        break;
      default:
        break;
    }
  }

  static _collectLoads(node, out) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'BufferLoadNode') { out.push({ buffer: node.buffer, indices: node.indices }); return; }
    if (node.a) FeatureExtractor._collectLoads(node.a, out);
    if (node.b) FeatureExtractor._collectLoads(node.b, out);
    if (node.args) for (const a of node.args) FeatureExtractor._collectLoads(a, out);
  }

  static _collectVars(node, set) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'VariableNode') { set.add(node.name); return; }
    if (node.a) FeatureExtractor._collectVars(node.a, set);
    if (node.b) FeatureExtractor._collectVars(node.b, set);
    if (node.args) for (const a of node.args) FeatureExtractor._collectVars(a, set);
    if (node.indices) for (const i of node.indices) FeatureExtractor._collectVars(i, set);
  }

  static _visitIterative(root, ctx) {
    const stack = [{ node: root, action: 'enter' }];
    while (stack.length > 0) {
      const { node, action } = stack.pop();
      if (!node) continue;

      if (action === 'leave_for') { ctx.currentDepth--; continue; }

      switch (node.type) {
        case 'ForNode': {
          ctx.numLoops++;
          ctx.currentDepth++;
          if (ctx.currentDepth > ctx.maxLoopDepth) ctx.maxLoopDepth = ctx.currentDepth;
          const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 1;
          ctx.loopExtents.push(extent);
          if (ctx.numLoops === 1) ctx.outermostExtent = extent;
          ctx.innermostExtent = extent;
          switch (node.kind) {
            case ForKind.PARALLEL: ctx.numParallelLoops++; break;
            case ForKind.VECTORIZED: ctx.numVectorizedLoops++; break;
            case ForKind.UNROLLED: ctx.numUnrolledLoops++; break;
            case ForKind.THREAD_BINDING:
              ctx.numThreadBound++;
              if (node.threadTag && node.threadTag.startsWith('threadIdx')) ctx.threadBlockSize *= extent;
              else if (node.threadTag && node.threadTag.startsWith('blockIdx')) ctx.gridSize *= extent;
              break;
            default: ctx.numSerialLoops++; break;
          }
          ctx.totalIterations *= extent;
          stack.push({ node: null, action: 'leave_for' });
          stack.push({ node: node.body, action: 'enter' });
          break;
        }
        case 'BlockNode':
          ctx.numBlocks++;
          if (node.initBody) { ctx.hasReduction = true; ctx.reductionDepth = ctx.currentDepth; }
          for (const r of node.reads) FeatureExtractor._visitBuffer(r.buffer, ctx);
          for (const w of node.writes) FeatureExtractor._visitBuffer(w.buffer, ctx);
          stack.push({ node: node.body, action: 'enter' });
          if (node.initBody) stack.push({ node: node.initBody, action: 'enter' });
          break;
        case 'SeqNode':
          for (let i = node.stmts.length - 1; i >= 0; i--) stack.push({ node: node.stmts[i], action: 'enter' });
          break;
        case 'AllocateNode':
          FeatureExtractor._visitBuffer(node.buffer, ctx);
          stack.push({ node: node.body, action: 'enter' });
          break;
        case 'IfThenElseNode':
          if (node.elseBody) stack.push({ node: node.elseBody, action: 'enter' });
          stack.push({ node: node.thenBody, action: 'enter' });
          break;
        case 'LetStmtNode':
          stack.push({ node: node.body, action: 'enter' });
          break;
        case 'BufferStoreNode':
          ctx.numBufferWrites++;
          FeatureExtractor._visitBuffer(node.buffer, ctx);
          FeatureExtractor._checkStride(node.buffer, node.indices, ctx);
          FeatureExtractor._visitExpr(node.value, ctx);
          break;
      }
    }
  }

  static _visitBuffer(buffer, ctx) {
    if (!buffer || ctx.buffersSeen.has(buffer)) return;
    ctx.buffersSeen.add(buffer);
    const bytes = buffer.sizeInBytes();
    if (bytes > 0) ctx.totalBufferBytes += bytes;
  }

  static _visitExpr(node, ctx) {
    if (!node) return;
    switch (node.type) {
      case 'MathOpNode':
        ctx.numMathOps++;
        FeatureExtractor._visitExpr(node.a, ctx);
        if (node.b) FeatureExtractor._visitExpr(node.b, ctx);
        break;
      case 'CallExternNode':
        ctx.numExternCalls++;
        for (const a of node.args) FeatureExtractor._visitExpr(a, ctx);
        break;
      case 'BufferLoadNode':
        ctx.numBufferReads++;
        FeatureExtractor._checkStride(node.buffer, node.indices, ctx);
        break;
      case 'CompareNode':
        ctx.numMathOps++;
        FeatureExtractor._visitExpr(node.a, ctx);
        FeatureExtractor._visitExpr(node.b, ctx);
        break;
      default: break;
    }
  }

  static _checkStride(buffer, indices, ctx) {
    if (!buffer || !indices || indices.length === 0) return;
    const lastIdx = indices[indices.length - 1];
    if (lastIdx && lastIdx.type === 'VariableNode') {
      ctx.strideOneAccesses++;
    } else {
      ctx.nonStrideOneAccesses++;
    }
  }
}
