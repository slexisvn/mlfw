import { ForKind } from '../ir/tensor/nodes.js';

export class ScheduleFeatures {
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
    this.tileFactors = raw.tileFactors || [];
  }

  toVector() {
    return [
      this.numLoops, this.numBlocks, this.totalIterations,
      this.maxLoopDepth, this.numParallelLoops, this.numVectorizedLoops,
      this.numUnrolledLoops, this.numThreadBound, this.numSerialLoops,
      this.totalBufferBytes, this.numBufferReads, this.numBufferWrites,
      this.numMathOps, this.numExternCalls, this.arithmeticIntensity,
      this.innermostExtent, this.outermostExtent,
      this.hasReduction ? 1 : 0, this.reductionDepth,
      this.threadBlockSize, this.gridSize,
      this.strideOneAccesses, this.nonStrideOneAccesses
    ];
  }

  static featureNames() {
    return [
      'numLoops', 'numBlocks', 'totalIterations',
      'maxLoopDepth', 'numParallelLoops', 'numVectorizedLoops',
      'numUnrolledLoops', 'numThreadBound', 'numSerialLoops',
      'totalBufferBytes', 'numBufferReads', 'numBufferWrites',
      'numMathOps', 'numExternCalls', 'arithmeticIntensity',
      'innermostExtent', 'outermostExtent',
      'hasReduction', 'reductionDepth',
      'threadBlockSize', 'gridSize',
      'strideOneAccesses', 'nonStrideOneAccesses'
    ];
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
      strideOneAccesses: 0, nonStrideOneAccesses: 0,
      tileFactors: []
    };

    FeatureExtractor._visit(primFunc.body, ctx);

    const bytes = ctx.totalBufferBytes;
    const ops = ctx.numMathOps + ctx.numExternCalls;
    ctx.arithmeticIntensity = bytes > 0 ? ops / bytes : 0;

    return new ScheduleFeatures(ctx);
  }

  static _visit(node, ctx) {
    if (!node) return;

    switch (node.type) {
      case 'ForNode':
        FeatureExtractor._visitFor(node, ctx);
        break;
      case 'BlockNode':
        FeatureExtractor._visitBlock(node, ctx);
        break;
      case 'SeqNode':
        for (const s of node.stmts) FeatureExtractor._visit(s, ctx);
        break;
      case 'AllocateNode':
        FeatureExtractor._visitBuffer(node.buffer, ctx);
        FeatureExtractor._visit(node.body, ctx);
        break;
      case 'IfThenElseNode':
        FeatureExtractor._visit(node.thenBody, ctx);
        if (node.elseBody) FeatureExtractor._visit(node.elseBody, ctx);
        break;
      case 'LetStmtNode':
        FeatureExtractor._visit(node.body, ctx);
        break;
      case 'BufferStoreNode':
        ctx.numBufferWrites++;
        FeatureExtractor._visitBuffer(node.buffer, ctx);
        FeatureExtractor._checkStride(node.buffer, node.indices, ctx);
        FeatureExtractor._visitExpr(node.value, ctx);
        break;
      default:
        break;
    }
  }

  static _visitFor(node, ctx) {
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
        if (node.threadTag && node.threadTag.startsWith('threadIdx')) {
          ctx.threadBlockSize *= extent;
        } else if (node.threadTag && node.threadTag.startsWith('blockIdx')) {
          ctx.gridSize *= extent;
        }
        break;
      default: ctx.numSerialLoops++; break;
    }

    ctx.totalIterations *= extent;
    FeatureExtractor._visit(node.body, ctx);
    ctx.currentDepth--;
  }

  static _visitBlock(node, ctx) {
    ctx.numBlocks++;
    if (node.initBody) {
      ctx.hasReduction = true;
      ctx.reductionDepth = ctx.currentDepth;
    }
    for (const r of node.reads) FeatureExtractor._visitBuffer(r.buffer, ctx);
    for (const w of node.writes) FeatureExtractor._visitBuffer(w.buffer, ctx);
    if (node.initBody) FeatureExtractor._visit(node.initBody, ctx);
    FeatureExtractor._visit(node.body, ctx);
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
