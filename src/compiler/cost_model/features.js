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

    FeatureExtractor._visitIterative(primFunc.body, ctx);

    const bytes = ctx.totalBufferBytes;
    const ops = ctx.numMathOps + ctx.numExternCalls;
    ctx.arithmeticIntensity = bytes > 0 ? ops / bytes : 0;

    return new ScheduleFeatures(ctx);
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
