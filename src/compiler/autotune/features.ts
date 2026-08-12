import { ForKind } from '../ir/tensor/nodes.js';
import type { TirNode, PrimFunc, ForNode, BlockNode, SeqNode, IfThenElseNode, AllocateNode, LetStmtNode, BufferStoreNode, BufferLoadNode, MathOpNode, CompareNode, CallExternNode, VariableNode, IntImmNode } from '../ir/tensor/nodes.js';
import type { Buffer } from '../ir/tensor/buffer.js';

type NodeSlots = Record<string, TirNode | TirNode[] | undefined>;

export type FeatureCounts = { math: number; extern: number };
export type AccessRef = { buffer: Buffer; indices: readonly TirNode[] };
export type RawFeatures = Record<string, number | boolean | Set<unknown> | number[]>;
export type ExtractCtx = {
  numLoops: number; numBlocks: number; totalIterations: number;
  maxLoopDepth: number; currentDepth: number;
  numParallelLoops: number; numVectorizedLoops: number;
  numUnrolledLoops: number; numThreadBound: number; numSerialLoops: number;
  totalBufferBytes: number; buffersSeen: Set<Buffer>;
  numBufferReads: number; numBufferWrites: number;
  numMathOps: number; numExternCalls: number;
  hasReduction: boolean; reductionDepth: number;
  threadBlockSize: number; gridSize: number;
  innermostExtent: number; outermostExtent: number;
  loopExtents: number[];
  strideOneAccesses: number; nonStrideOneAccesses: number;
  arithmeticIntensity?: number;
};
type VisitFrame = { node: TirNode | null; action: string };

export const STATEMENT_FEATURE_SCHEMA = [
  'iterCount', 'depth',
  'parallelLoops', 'vectorizedLoops', 'unrolledLoops', 'threadBoundLoops', 'serialLoops',
  'threadBlockSize', 'gridSize', 'underReduction',
  'numMathOps', 'numExternCalls', 'numReads', 'numWrites',
  'stride1Accesses', 'stridedAccesses', 'reuseCount', 'touchedBytes',
  'arithmeticIntensity', 'vectorized', 'parallelized', 'innermostExtent'
];

export type ScheduleFeatureSet = {
  numLoops: number; numBlocks: number; totalIterations: number; maxLoopDepth: number;
  numParallelLoops: number; numVectorizedLoops: number; numUnrolledLoops: number;
  numThreadBound: number; numSerialLoops: number; totalBufferBytes: number;
  numBufferReads: number; numBufferWrites: number; numMathOps: number; numExternCalls: number;
  arithmeticIntensity: number; innermostExtent: number; outermostExtent: number;
  hasReduction: boolean; reductionDepth: number; threadBlockSize: number; gridSize: number;
  strideOneAccesses: number; nonStrideOneAccesses: number;
};

export class ScheduleFeatures implements ScheduleFeatureSet {
  numLoops!: number; numBlocks!: number; totalIterations!: number; maxLoopDepth!: number;
  numParallelLoops!: number; numVectorizedLoops!: number; numUnrolledLoops!: number;
  numThreadBound!: number; numSerialLoops!: number; totalBufferBytes!: number;
  numBufferReads!: number; numBufferWrites!: number; numMathOps!: number; numExternCalls!: number;
  arithmeticIntensity!: number; innermostExtent!: number; outermostExtent!: number;
  hasReduction!: boolean; reductionDepth!: number; threadBlockSize!: number; gridSize!: number;
  strideOneAccesses!: number; nonStrideOneAccesses!: number;

  constructor(raw: ExtractCtx) {
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
  static extract(primFunc: PrimFunc): ScheduleFeatures {
    const ctx: ExtractCtx = {
      numLoops: 0, numBlocks: 0, totalIterations: 1,
      maxLoopDepth: 0, currentDepth: 0,
      numParallelLoops: 0, numVectorizedLoops: 0,
      numUnrolledLoops: 0, numThreadBound: 0, numSerialLoops: 0,
      totalBufferBytes: 0, buffersSeen: new Set<Buffer>(),
      numBufferReads: 0, numBufferWrites: 0,
      numMathOps: 0, numExternCalls: 0,
      hasReduction: false, reductionDepth: 0,
      threadBlockSize: 1, gridSize: 1,
      innermostExtent: 0, outermostExtent: 0,
      loopExtents: [] as number[],
      strideOneAccesses: 0, nonStrideOneAccesses: 0
    };

    FeatureExtractor._visitIterative(primFunc.body, ctx);

    const bytes = ctx.totalBufferBytes;
    const ops = ctx.numMathOps + ctx.numExternCalls;
    ctx.arithmeticIntensity = bytes > 0 ? ops / bytes : 0;

    return new ScheduleFeatures(ctx);
  }

  static extractStatements(primFunc: PrimFunc): number[][] {
    const out: number[][] = [];
    const loopStack: ForNode[] = [];
    let reduction = 0;
    const stack: VisitFrame[] = [{ node: primFunc.body, action: 'enter' }];
    while (stack.length > 0) {
      const { node, action } = stack.pop() as VisitFrame;
      if (!node) continue;
      if (action === 'leaveFor') { loopStack.pop(); continue; }
      if (action === 'leaveBlock') { reduction--; continue; }
      switch (node.type) {
        case 'ForNode':
          loopStack.push(node as ForNode);
          stack.push({ node: null, action: 'leaveFor' });
          stack.push({ node: (node as ForNode).body, action: 'enter' });
          break;
        case 'BlockNode': {
          const b = node as BlockNode;
          if (b.initBody) {
            reduction++;
            stack.push({ node: null, action: 'leaveBlock' });
            stack.push({ node: b.initBody, action: 'enter' });
          }
          stack.push({ node: b.body, action: 'enter' });
        }
          break;
        case 'SeqNode': {
          const seq = node as SeqNode;
          for (let i = seq.stmts.length - 1; i >= 0; i--) stack.push({ node: seq.stmts[i], action: 'enter' });
        }
          break;
        case 'AllocateNode':
        case 'LetStmtNode':
          stack.push({ node: (node as AllocateNode | LetStmtNode).body, action: 'enter' });
          break;
        case 'IfThenElseNode': {
          const ite = node as IfThenElseNode;
          if (ite.elseBody) stack.push({ node: ite.elseBody, action: 'enter' });
          stack.push({ node: ite.thenBody, action: 'enter' });
        }
          break;
        case 'BufferStoreNode':
          out.push(FeatureExtractor._statementVector(node as BufferStoreNode, loopStack, reduction));
          break;
      }
    }
    return out;
  }

  static _statementVector(store: BufferStoreNode, loopStack: readonly ForNode[], reduction: number): number[] {
    let iterCount = 1, par = 0, vec = 0, unr = 0, thr = 0, ser = 0, tbs = 1, grid = 1;
    for (const f of loopStack) {
      const ext = f.extent && f.extent.type === 'IntImmNode' ? (f.extent as IntImmNode).value : 1;
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
    const arith: FeatureCounts = { math: 0, extern: 0 };
    FeatureExtractor._countExpr(store.value, arith);
    const loopVarNames = loopStack.map(f => f.loopVar.name);
    const accesses: AccessRef[] = [{ buffer: store.buffer, indices: store.indices }];
    FeatureExtractor._collectLoads(store.value, accesses);
    let stride1 = 0, strided = 0, reuse = 0, touched = 0;
    for (const acc of accesses) {
      const accBytes = acc.buffer && acc.buffer.sizeInBytes ? acc.buffer.sizeInBytes() : 0;
      if (accBytes > 0) touched += accBytes;
      const last = acc.indices && acc.indices.length > 0 ? acc.indices[acc.indices.length - 1] : null;
      if (last && last.type === 'VariableNode') stride1++;
      else strided++;
      const used = new Set<string>();
      if (acc.indices) for (const idx of acc.indices) FeatureExtractor._collectVars(idx, used);
      for (const name of loopVarNames) if (!used.has(name)) reuse++;
    }
    const innermost = loopStack.length > 0 ? loopStack[loopStack.length - 1] : null;
    const innermostExtent = innermost && innermost.extent && innermost.extent.type === 'IntImmNode' ? (innermost.extent as IntImmNode).value : 0;
    const fields: Record<string, number> = {
      iterCount, depth: loopStack.length,
      parallelLoops: par, vectorizedLoops: vec, unrolledLoops: unr,
      threadBoundLoops: thr, serialLoops: ser,
      threadBlockSize: tbs, gridSize: grid,
      underReduction: reduction > 0 ? 1 : 0,
      numMathOps: arith.math, numExternCalls: arith.extern,
      numReads: accesses.length - 1, numWrites: 1,
      stride1Accesses: stride1, stridedAccesses: strided,
      reuseCount: reuse, touchedBytes: touched,
      arithmeticIntensity: touched > 0 ? (arith.math + arith.extern) / touched : 0,
      vectorized: vec > 0 ? 1 : 0,
      parallelized: (par + thr) > 0 ? 1 : 0,
      innermostExtent
    };
    return STATEMENT_FEATURE_SCHEMA.map(n => fields[n] || 0);
  }

  static _countExpr(node: TirNode | null | undefined, acc: FeatureCounts): void {
    if (!node || typeof node !== 'object') return;
    switch (node.type) {
      case 'MathOpNode':
        acc.math++;
        FeatureExtractor._countExpr((node as MathOpNode).a, acc);
        if ((node as MathOpNode).b) FeatureExtractor._countExpr((node as MathOpNode).b, acc);
        break;
      case 'CompareNode':
        acc.math++;
        FeatureExtractor._countExpr((node as CompareNode).a, acc);
        FeatureExtractor._countExpr((node as CompareNode).b, acc);
        break;
      case 'CallExternNode':
        acc.extern++;
        for (const a of (node as CallExternNode).args) FeatureExtractor._countExpr(a, acc);
        break;
      default:
        break;
    }
  }

  static _collectLoads(node: TirNode | null | undefined, out: AccessRef[]): void {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'BufferLoadNode') { const ld = node as BufferLoadNode; out.push({ buffer: ld.buffer, indices: ld.indices }); return; }
    const slots = node as unknown as NodeSlots;
    if (slots.a) FeatureExtractor._collectLoads(slots.a as TirNode, out);
    if (slots.b) FeatureExtractor._collectLoads(slots.b as TirNode, out);
    if (slots.args) for (const a of slots.args as TirNode[]) FeatureExtractor._collectLoads(a, out);
  }

  static _collectVars(node: TirNode | null | undefined, set: Set<string>): void {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'VariableNode') { set.add((node as VariableNode).name); return; }
    const slots = node as unknown as NodeSlots;
    if (slots.a) FeatureExtractor._collectVars(slots.a as TirNode, set);
    if (slots.b) FeatureExtractor._collectVars(slots.b as TirNode, set);
    if (slots.args) for (const a of slots.args as TirNode[]) FeatureExtractor._collectVars(a, set);
    if (slots.indices) for (const i of slots.indices as TirNode[]) FeatureExtractor._collectVars(i, set);
  }

  static _visitIterative(root: TirNode | null | undefined, ctx: ExtractCtx): void {
    const stack = [{ node: root, action: 'enter' }];
    while (stack.length > 0) {
      const { node, action } = stack.pop() as VisitFrame;
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
        case 'SeqNode': {
          const seq = node as SeqNode;
          for (let i = seq.stmts.length - 1; i >= 0; i--) stack.push({ node: seq.stmts[i], action: 'enter' });
        }
          break;
        case 'AllocateNode':
          FeatureExtractor._visitBuffer(node.buffer, ctx);
          stack.push({ node: node.body, action: 'enter' });
          break;
        case 'IfThenElseNode': {
          const ite = node as IfThenElseNode;
          if (ite.elseBody) stack.push({ node: ite.elseBody, action: 'enter' });
          stack.push({ node: ite.thenBody, action: 'enter' });
        }
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

  static _visitBuffer(buffer: Buffer | null | undefined, ctx: ExtractCtx): void {
    if (!buffer || ctx.buffersSeen.has(buffer)) return;
    ctx.buffersSeen.add(buffer);
    const bytes = buffer.sizeInBytes();
    if (bytes > 0) ctx.totalBufferBytes += bytes;
  }

  static _visitExpr(node: TirNode | null | undefined, ctx: ExtractCtx): void {
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

  static _checkStride(buffer: Buffer | null | undefined, indices: readonly TirNode[] | null | undefined, ctx: ExtractCtx): void {
    if (!buffer || !indices || indices.length === 0) return;
    let lastIdx = indices[indices.length - 1];
    while (lastIdx && lastIdx.type === 'MathOpNode' && lastIdx.b && lastIdx.b.type === 'IntImmNode') {
      if (lastIdx.op === '+' && lastIdx.b.value === 0) lastIdx = lastIdx.a;
      else if (lastIdx.op === '*' && lastIdx.b.value === 1) lastIdx = lastIdx.a;
      else break;
    }
    if (lastIdx && lastIdx.type === 'VariableNode') {
      ctx.strideOneAccesses++;
    } else {
      ctx.nonStrideOneAccesses++;
    }
  }
}
