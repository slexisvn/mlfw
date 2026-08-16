
import { TargetKind } from '../../backend/target.js';
import { ForKind } from '../ir/tensor/nodes.js';
import { some as irSome, collect as irCollect } from '../ir/ir_visitor.js';
import { reductionLoopVars } from './legality.js';
import type { IRNode } from '../ir/ir_visitor.js';
import type { TirNode, PrimFunc, ForNode, BlockNode, SeqNode, IfThenElseNode, AllocateNode, LetStmtNode, IntImmNode } from '../ir/tensor/nodes.js';
import type { Schedule } from './schedule.js';
import type { ScheduleTarget } from './gpu_matmul_schedule.js';
export type ScheduleExplainTrace = { explainsEnabled: boolean; explain(category: string, subject: string, decision: string, reason: string, data: Record<string, unknown>): void };

export type BlockClassification = {
  loopCount: number;
  hasReduction: boolean;
  reductionLoopVars: Set<string>;
  readBuffers: string[];
  writeBuffers: string[];
  loops: ForNode[];
};

type ClassifyCache = Map<string, BlockClassification>;
type CollectFrame = { node: TirNode | null | undefined; loops: ForNode[] };

export class ScheduleRule {
  name: string;

  constructor(name: string) {
    this.name = name;
  }

  matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    throw new Error('ScheduleRule.matches must be implemented');
  }

  apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    throw new Error('ScheduleRule.apply must be implemented');
  }
}

const _classifyCacheByFunc = new WeakMap<PrimFunc, ClassifyCache>();

export function classifyBlock(primFunc: PrimFunc, blockName: string): BlockClassification | null {
  let cache = _classifyCacheByFunc.get(primFunc);
  if (cache && cache.has(blockName)) return cache.get(blockName) as BlockClassification;
  if (!cache) {
    cache = new Map();
    _classifyCacheByFunc.set(primFunc, cache);
  }
  collectBlockInfo(primFunc.body, cache, []);
  return cache.has(blockName) ? cache.get(blockName) as BlockClassification : null;
}

export function invalidateClassifyCache(primFunc: PrimFunc | null | undefined): void {
  if (primFunc) _classifyCacheByFunc.delete(primFunc);
}

function invalidateClassifyBlock(primFunc: PrimFunc, blockName: string): void {
  const cache = _classifyCacheByFunc.get(primFunc);
  if (cache) cache.delete(blockName);
}

function collectBlockInfo(root: TirNode, result: ClassifyCache, initialLoopStack: readonly ForNode[]): void {
  const stack: CollectFrame[] = [{ node: root, loops: [...initialLoopStack] }];
  while (stack.length > 0) {
    const { node, loops } = stack.pop() as CollectFrame;
    if (!node) continue;
    if (node.type === 'ForNode') {
      const f = node as ForNode;
      stack.push({ node: f.body, loops: [...loops, f] });
    } else if (node.type === 'BlockNode') {
      const b = node as BlockNode;
      if (!result.has(b.name)) {
        const reduction = reductionLoopVars(b);
        result.set(b.name, {
          loopCount: loops.length,
          hasReduction: b.initBody !== null || reduction.size > 0,
          reductionLoopVars: reduction,
          readBuffers: b.reads.map(r => r.buffer.name),
          writeBuffers: b.writes.map(r => r.buffer.name),
          loops: [...loops]
        });
      }
      stack.push({ node: b.body, loops });
    } else if (node.type === 'SeqNode') {
      const seq = node as SeqNode;
      for (let i = seq.stmts.length - 1; i >= 0; i--) stack.push({ node: seq.stmts[i], loops });
    } else if (node.type === 'IfThenElseNode') {
      const ite = node as IfThenElseNode;
      if (ite.elseBody) stack.push({ node: ite.elseBody, loops });
      stack.push({ node: ite.thenBody, loops });
    } else if (node.type === 'AllocateNode') {
      stack.push({ node: (node as AllocateNode).body, loops });
    } else if (node.type === 'LetStmtNode') {
      stack.push({ node: (node as LetStmtNode).body, loops });
    }
  }
}

function isMatmulShape(info: BlockClassification): boolean {
  return info.hasReduction && info.readBuffers.length === 2 &&
         info.writeBuffers.length === 1 && info.loopCount >= 3;
}

function blockHasNonConstExtent(primFunc: PrimFunc, blockName: string): boolean {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return false;
  return info.loops.some(l => l.extent && l.extent.type !== 'IntImmNode');
}

function hasMultipleBlocks(loop: ForNode): boolean {
  let count = 0;
  const stack: (TirNode | null | undefined)[] = [loop.body];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BlockNode') { count++; if (count > 1) return true; }
    if (n.type === 'ForNode') stack.push((n as ForNode).body);
    if (n.type === 'SeqNode') for (const st of (n as SeqNode).stmts) stack.push(st);
  }
  return false;
}

export class ElementwiseCPURule extends ScheduleRule {
  constructor() {
    super('elementwise_cpu');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (info.hasReduction || info.loopCount < 1) return false;
    if (info.loops.length > 0 && hasMultipleBlocks(info.loops[0])) return false;
    let totalIters = 1;
    for (const l of info.loops) {
      const e = l.extent && l.extent.type === 'IntImmNode' ? l.extent.value : 1;
      totalIters *= e;
    }
    return totalIters >= target.numCores * target.vectorWidth;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length === 0) return;

    if (loops.length === 1) {
      const extent = loops[0].extent;
      if (extent.type === 'IntImmNode' && extent.value >= target.vectorWidth * 2) {
        const [outer, inner] = schedule.split(loops[0], target.vectorWidth);
        schedule.parallelize(outer);
        schedule.vectorize(inner);
        return;
      }
      schedule.parallelize(loops[0]);
      return;
    }

    schedule.parallelize(loops[0]);

    const innermost = loops[loops.length - 1];
    const extent = innermost.extent;
    if (extent.type === 'IntImmNode' && extent.value >= target.vectorWidth) {
      if (extent.value % target.vectorWidth === 0) {
        const [_, vi] = schedule.split(innermost, target.vectorWidth);
        schedule.vectorize(vi);
      }
    }
  }
}

export function primFuncHasReduction(primFunc: PrimFunc): boolean {
  return irSome(primFunc.body, (n: IRNode) => n.type === 'BlockNode' && ((n as BlockNode).initBody !== null || reductionLoopVars(n as BlockNode).size > 0), { kinds: 'stmt' });
}

export function primFuncHasRecurrence(primFunc: PrimFunc): boolean {
  return irSome(primFunc.body, (n: IRNode) => n.type === 'ForNode' && (n as ForNode).kind === ForKind.RECURRENCE, { kinds: 'stmt' });
}

function bindFusedSpatialGPU(schedule: Schedule, fused: ForNode, target: ScheduleTarget): void {
  const blockSize = Math.min(target.maxThreadsPerBlock, 256);
  const extent = fused.extent;
  if (extent.type === 'IntImmNode' && extent.value > blockSize) {
    const [outer, tx] = schedule.split(fused, blockSize);
    schedule.bindThread(tx, 'threadIdx.x');
    if (!primFuncHasRecurrence(schedule.func)) schedule.bindThread(outer, 'blockIdx.x');
  } else {
    schedule.bindThread(fused, 'threadIdx.x');
  }
}

export class ElementwiseGPURule extends ScheduleRule {
  constructor() {
    super('elementwise_gpu');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (!target.isGPU()) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return !info.hasReduction && info.loopCount >= 1;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length === 0) return;

    let fusedLoop = loops[0];
    for (let i = 1; i < loops.length; i++) {
      const currentLoops = schedule.getLoops(blockName) as ForNode[];
      const nextLoop = currentLoops.find(l => l.loopVar.name === loops[i].loopVar.name);
      if (nextLoop && findDirectChild(fusedLoop, nextLoop)) {
        fusedLoop = schedule.fuseLoops(fusedLoop, nextLoop);
      }
    }

    const extent = fusedLoop.extent;
    if (extent.type !== 'IntImmNode') {
      schedule.bindThread(fusedLoop, 'threadIdx.x');
      return;
    }

    const totalElements = extent.value;
    const singleBlockCap = Math.min(target.maxThreadsPerBlock, 1024);
    if (primFuncHasReduction(schedule.func) && totalElements <= singleBlockCap) {
      schedule.bindThread(fusedLoop, 'threadIdx.x');
      return;
    }

    bindFusedSpatialGPU(schedule, fusedLoop, target);
  }
}

export class ReductionCPURule extends ScheduleRule {
  constructor() {
    super('reduction_cpu');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length === 0) return;

    const spatialLoops = [];
    const reduceLoops = [];

    const info = classifyBlock(schedule.func, blockName);
    for (const loop of loops) {
      if (info && isReductionLoop(loop, info)) {
        reduceLoops.push(loop);
      } else {
        spatialLoops.push(loop);
      }
    }

    if (spatialLoops.length > 0) {
      schedule.parallelize(spatialLoops[0]);
    }
  }
}

export class ReductionGPURule extends ScheduleRule {
  constructor() {
    super('reduction_gpu');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (!target.isGPU()) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length === 0) return;

    const spatialLoops: ForNode[] = [];
    const info = classifyBlock(schedule.func, blockName);
    for (const loop of loops) {
      if (!info || !isReductionLoop(loop, info)) {
        spatialLoops.push(loop);
      }
    }

    if (spatialLoops.length === 0) return;

    let fusedSpatial = spatialLoops[0];
    for (let i = 1; i < spatialLoops.length; i++) {
      const currentLoops = schedule.getLoops(blockName) as ForNode[];
      const next = currentLoops.find(l => l.loopVar.name === spatialLoops[i].loopVar.name);
      if (next && findDirectChild(fusedSpatial, next)) {
        fusedSpatial = schedule.fuseLoops(fusedSpatial, next);
      }
    }

    bindFusedSpatialGPU(schedule, fusedSpatial, target);
  }
}

export class MatmulTiledCPURule extends ScheduleRule {
  constructor() {
    super('matmul_tiled_cpu');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (!isMatmulShape(info)) return false;
    const cacheBytes = target.l1CacheBytes || 32768;
    const tileDim = Math.max(8, Math.min(64, Math.floor(Math.sqrt(cacheBytes / 4))));
    const maxExtent = info.loops.reduce((m, l) => {
      const e = l.extent && l.extent.type === 'IntImmNode' ? (l.extent as IntImmNode).value : 0;
      return e > m ? e : m;
    }, 0);
    return maxExtent >= tileDim;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length < 3) return;

    const cacheBytes = target.l1CacheBytes || 32768;
    const tileDim = Math.max(8, Math.min(64, Math.floor(Math.sqrt(cacheBytes / 4))));

    const tileIndices: number[] = [];
    const tileSizes: number[] = [];
    for (let i = 0; i < Math.min(2, loops.length); i++) {
      const ext = loops[i].extent.type === 'IntImmNode' ? (loops[i].extent as IntImmNode).value : null;
      if (ext && ext >= tileDim) {
        tileIndices.push(i);
        tileSizes.push(tileDim);
      }
    }

    if (tileIndices.length === 0) return;

    const { outerLoops } = schedule.tile(blockName, tileIndices, tileSizes);
    if (outerLoops.length > 0) schedule.parallelize(outerLoops[0]);
  }
}

export class MatmulTiledGPURule extends ScheduleRule {
  constructor() {
    super('matmul_tiled_gpu');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (!target.isGPU()) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (!isMatmulShape(info)) return false;
    const smemBytes = target.sharedMemoryBytes || 49152;
    const bytesPerTile = 4 * 2;
    const tileDim = Math.max(16, Math.min(128, Math.floor(Math.sqrt(smemBytes / bytesPerTile))));
    const maxExtent = info.loops.reduce((m, l) => {
      const e = l.extent && l.extent.type === 'IntImmNode' ? (l.extent as IntImmNode).value : 0;
      return e > m ? e : m;
    }, 0);
    return maxExtent >= tileDim;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length < 3) return;

    const info = classifyBlock(schedule.func, blockName);
    const spatial = loops.filter(l => !info || !isReductionLoop(l, info));
    if (spatial.length === 0) return;

    let fused = spatial[0];
    for (let i = 1; i < spatial.length; i++) {
      const current = schedule.getLoops(blockName) as ForNode[];
      const next = current.find(l => l.loopVar.name === spatial[i].loopVar.name);
      if (next && findDirectChild(fused, next)) fused = schedule.fuseLoops(fused, next);
    }

    bindFusedSpatialGPU(schedule, fused, target);
  }
}

export class ElementwiseWasmRule extends ScheduleRule {
  constructor() {
    super('elementwise_wasm');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (target.kind !== TargetKind.WASM) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (info.hasReduction || info.loopCount < 1) return false;
    if (info.loops.length > 0 && hasMultipleBlocks(info.loops[0])) return false;
    return true;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length === 0) return;

    const numCores = target.numCores || 1;
    const vectorWidth = target.vectorWidth || 4;

    if (numCores > 1 && loops.length >= 1) {
      const outerExtent = loops[0].extent;
      const outerSize = outerExtent && outerExtent.type === 'IntImmNode' ? outerExtent.value : 0;
      if (outerSize >= numCores * 4) {
        if (loops.length === 1 && target.supportsSimd && target.supportsSimd()
            && outerSize >= vectorWidth * 2 && outerSize % vectorWidth === 0) {
          const [outer, inner] = schedule.split(loops[0], vectorWidth);
          schedule.parallelize(outer);
          schedule.vectorize(inner);
          return;
        }
        schedule.parallelize(loops[0]);
        if (loops.length > 1) {
          const innermost = loops[loops.length - 1];
          const extent = innermost.extent;
          if (extent.type === 'IntImmNode' && extent.value >= vectorWidth) {
            if (extent.value % vectorWidth === 0) {
              const [, vi] = schedule.split(innermost, vectorWidth);
              schedule.vectorize(vi);
            }
          }
        }
        return;
      }
    }

    const innermost = loops[loops.length - 1];
    const extent = innermost.extent;

    if (extent.type === 'IntImmNode' && extent.value >= vectorWidth * 2) {
      const [outer, inner] = schedule.split(innermost, vectorWidth);
      schedule.vectorize(inner);
      return;
    }

    if (extent.type === 'IntImmNode' && extent.value >= vectorWidth) {
      schedule.vectorize(innermost);
    }
  }
}

export class ReductionWasmRule extends ScheduleRule {
  constructor() {
    super('reduction_wasm');
  }

  override matches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): boolean {
    if (target.kind !== TargetKind.WASM) return false;
    const hasParallel = target.numCores > 1;
    const hasSimd = target.supportsSimd && target.supportsSimd();
    if (!hasParallel && !hasSimd) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction && info.loopCount >= 2;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length < 2) return;

    const info = classifyBlock(schedule.func, blockName);
    const spatialLoops = [];
    const reductionLoops = [];
    for (const loop of loops) {
      if (!info || !isReductionLoop(loop, info)) {
        spatialLoops.push(loop);
      } else {
        reductionLoops.push(loop);
      }
    }

    if (spatialLoops.length > 0) {
      const outerExtent = spatialLoops[0].extent;
      const outerSize = outerExtent && outerExtent.type === 'IntImmNode' ? outerExtent.value : 0;
      if (outerSize >= (target.numCores || 1) * 4) {
        schedule.parallelize(spatialLoops[0]);
      }
    }

    if (target.supportsSimd && target.supportsSimd() && reductionLoops.length > 0) {
      const redLoop = reductionLoops[reductionLoops.length - 1];
      const redExtent = redLoop.extent && redLoop.extent.type === 'IntImmNode' ? redLoop.extent.value : 0;
      if (redExtent >= target.vectorWidth * 2) {
        schedule.vectorize(redLoop);
      }
    }
  }
}

export class FallbackRule extends ScheduleRule {
  constructor() {
    super('fallback');
  }

  override matches(): boolean {
    return true;
  }

  override apply(schedule: Schedule, blockName: string, target: ScheduleTarget): void {
    const loops = schedule.getLoops(blockName) as ForNode[];
    if (loops.length === 0) return;
    if (target.isCPU() && loops.length >= 1) {
      schedule.parallelize(loops[0]);
    }
  }
}

export function isReductionLoop(loop: ForNode, blockInfo: BlockClassification): boolean {
  if (!blockInfo.reductionLoopVars) return false;
  return blockInfo.reductionLoopVars.has(loop.loopVar.name);
}

function findDirectChild(parent: ForNode, child: ForNode): boolean {
  return parent.body === child;
}

export class SchedulePolicy {
  target: ScheduleTarget;
  rules: ScheduleRule[];
  trace: ScheduleExplainTrace | null;

  constructor(target: ScheduleTarget, rules: ScheduleRule[] | null = null, trace: ScheduleExplainTrace | null = null) {
    this.target = target;
    this.rules = rules || SchedulePolicy.defaultRules();
    this.trace = trace;
  }

  static defaultRules(): ScheduleRule[] {
    return [
      new MatmulTiledCPURule(),
      new MatmulTiledGPURule(),
      new ReductionCPURule(),
      new ReductionGPURule(),
      new ReductionWasmRule(),
      new ElementwiseCPURule(),
      new ElementwiseGPURule(),
      new ElementwiseWasmRule(),
      new FallbackRule()
    ];
  }

  selectRule(primFunc: PrimFunc, blockName: string): ScheduleRule | null {
    for (const rule of this.rules) {
      if (rule.matches(primFunc, blockName, this.target)) {
        return rule;
      }
    }
    return null;
  }

  applyToBlock(schedule: Schedule, blockName: string): string | null {
    if (this.target.isGPU() && blockHasNonConstExtent(schedule.func, blockName)) {
      this._explain(blockName, 'none', 'block has dynamic loop extents; runs sequentially (no dynamic grid)');
      return null;
    }
    const rule = this.selectRule(schedule.func, blockName);
    if (rule) {
      try {
        rule.apply(schedule, blockName, this.target);
      } catch (e) {
        invalidateClassifyBlock(schedule.func, blockName);
        this._explain(blockName, 'none', `rule '${rule.name}' rejected: ${(e as Error).message}`);
        return null;
      }
      invalidateClassifyBlock(schedule.func, blockName);
      this._explain(blockName, rule.name, `matched rule '${rule.name}' for ${this.target.name}`);
      return rule.name;
    }
    this._explain(blockName, 'none', 'no schedule rule matched; runs sequentially');
    return null;
  }

  _explain(blockName: string, decision: string, reason: string): void {
    if (this.trace && this.trace.explainsEnabled) {
      this.trace.explain('schedule', blockName, decision, reason, { target: this.target.name });
    }
  }

  applyToAllBlocks(schedule: Schedule): Map<string, string> {
    invalidateClassifyCache(schedule.func);
    const blocks = collectAllBlockNames(schedule.func.body);
    const seen = new Set<string>();
    const applied = new Map<string, string>();
    for (const name of blocks) {
      if (seen.has(name)) continue;
      seen.add(name);
      const ruleName = this.applyToBlock(schedule, name);
      if (ruleName) applied.set(name, ruleName);
    }
    return applied;
  }
}

function collectAllBlockNames(root: TirNode): string[] {
  return irCollect(root, (n: IRNode) => n.type === 'BlockNode', { kinds: 'stmt' }).map((n) => (n as BlockNode).name);
}
