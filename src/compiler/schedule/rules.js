
import { TargetKind } from '../../backend/target.js';
import { ForKind } from '../ir/tensor/nodes.js';
import { some as irSome, collect as irCollect } from '../ir/ir_visitor.js';

export class ScheduleRule {
  constructor(name) {
    this.name = name;
  }

  matches(primFunc, blockName, target) {
    throw new Error('ScheduleRule.matches must be implemented');
  }

  apply(schedule, blockName, target) {
    throw new Error('ScheduleRule.apply must be implemented');
  }
}

const _classifyCacheByFunc = new WeakMap();

export function classifyBlock(primFunc, blockName) {
  let cache = _classifyCacheByFunc.get(primFunc);
  if (!cache) {
    cache = new Map();
    collectBlockInfo(primFunc.body, cache, []);
    _classifyCacheByFunc.set(primFunc, cache);
  }
  return cache.get(blockName) || null;
}

function invalidateClassifyCache(primFunc) {
  if (primFunc) _classifyCacheByFunc.delete(primFunc);
}

function collectVarNames(node, out) {
  const stack = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object') continue;
    if (n.type === 'VariableNode') { out.add(n.name); continue; }
    for (const k of ['a', 'b', 'condition', 'thenBody', 'elseBody', 'expr', 'value', 'offsetExpr', 'extent']) {
      if (n[k]) stack.push(n[k]);
    }
    if (n.indices) for (const i of n.indices) stack.push(i);
    if (n.args) for (const a of n.args) stack.push(a);
  }
}

function computeReductionLoopVars(blockNode) {
  const writeBufs = new Set((blockNode.writes || []).map(w => w.buffer && w.buffer.name));
  const writeIdxVars = new Set();
  const stack = [blockNode.body, blockNode.initBody];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BufferStoreNode' && n.buffer && writeBufs.has(n.buffer.name)) {
      for (const idx of n.indices) collectVarNames(idx, writeIdxVars);
    }
    if (n.body) stack.push(n.body);
    if (n.stmts) for (const s of n.stmts) stack.push(s);
    if (n.thenBody) stack.push(n.thenBody);
    if (n.elseBody) stack.push(n.elseBody);
    if (n.value) stack.push(n.value);
  }
  const red = new Set();
  for (const iv of (blockNode.iterVars || [])) {
    const bindVars = new Set();
    collectVarNames(iv.binding, bindVars);
    const isSpatial = (iv.iterVar && writeIdxVars.has(iv.iterVar.name))
      || [...bindVars].some(v => writeIdxVars.has(v));
    if (!isSpatial) for (const v of bindVars) red.add(v);
  }
  return red;
}

function collectBlockInfo(root, result, initialLoopStack) {
  const stack = [{ node: root, loops: [...initialLoopStack] }];
  while (stack.length > 0) {
    const { node, loops } = stack.pop();
    if (!node) continue;
    if (node.type === 'ForNode') {
      stack.push({ node: node.body, loops: [...loops, node] });
    } else if (node.type === 'BlockNode') {
      const reductionLoopVars = computeReductionLoopVars(node);
      result.set(node.name, {
        loopCount: loops.length,
        hasReduction: node.initBody !== null || reductionLoopVars.size > 0,
        reductionLoopVars,
        readBuffers: node.reads.map(r => r.buffer.name),
        writeBuffers: node.writes.map(r => r.buffer.name),
        loops: [...loops]
      });
      stack.push({ node: node.body, loops });
    } else if (node.type === 'SeqNode') {
      for (let i = node.stmts.length - 1; i >= 0; i--) stack.push({ node: node.stmts[i], loops });
    } else if (node.type === 'IfThenElseNode') {
      if (node.elseBody) stack.push({ node: node.elseBody, loops });
      stack.push({ node: node.thenBody, loops });
    } else if (node.type === 'AllocateNode') {
      stack.push({ node: node.body, loops });
    } else if (node.type === 'LetStmtNode') {
      stack.push({ node: node.body, loops });
    }
  }
}

function isMatmulShape(info) {
  return info.hasReduction && info.readBuffers.length === 2 &&
         info.writeBuffers.length === 1 && info.loopCount >= 3;
}

function blockHasNonConstExtent(primFunc, blockName) {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return false;
  return info.loops.some(l => l.extent && l.extent.type !== 'IntImmNode');
}

function hasMultipleBlocks(loop) {
  let count = 0;
  const stack = [loop.body];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BlockNode') { count++; if (count > 1) return true; }
    if (n.type === 'ForNode') stack.push(n.body);
    if (n.type === 'SeqNode') for (const s of n.stmts) stack.push(s);
  }
  return false;
}

export class ElementwiseCPURule extends ScheduleRule {
  constructor() {
    super('elementwise_cpu');
  }

  matches(primFunc, blockName, target) {
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

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
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

export function primFuncHasReduction(primFunc) {
  return irSome(primFunc.body, (n) => n.type === 'BlockNode' && (n.initBody !== null || computeReductionLoopVars(n).size > 0), { kinds: 'stmt' });
}

export function primFuncHasRecurrence(primFunc) {
  return irSome(primFunc.body, (n) => n.type === 'ForNode' && n.kind === ForKind.RECURRENCE, { kinds: 'stmt' });
}

function bindFusedSpatialGPU(schedule, fused, target) {
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

  matches(primFunc, blockName, target) {
    if (!target.isGPU()) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return !info.hasReduction && info.loopCount >= 1;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    let fusedLoop = loops[0];
    for (let i = 1; i < loops.length; i++) {
      const currentLoops = schedule.getLoops(blockName);
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

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
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

  matches(primFunc, blockName, target) {
    if (!target.isGPU()) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    const spatialLoops = [];
    const info = classifyBlock(schedule.func, blockName);
    for (const loop of loops) {
      if (!info || !isReductionLoop(loop, info)) {
        spatialLoops.push(loop);
      }
    }

    if (spatialLoops.length === 0) return;

    let fusedSpatial = spatialLoops[0];
    for (let i = 1; i < spatialLoops.length; i++) {
      const currentLoops = schedule.getLoops(blockName);
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

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (!isMatmulShape(info)) return false;
    const cacheBytes = target.l1CacheBytes || 32768;
    const tileDim = Math.max(8, Math.min(64, Math.floor(Math.sqrt(cacheBytes / 4))));
    const maxExtent = info.loops.reduce((m, l) => {
      const e = l.extent && l.extent.type === 'IntImmNode' ? l.extent.value : 0;
      return e > m ? e : m;
    }, 0);
    return maxExtent >= tileDim;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const cacheBytes = target.l1CacheBytes || 32768;
    const tileDim = Math.max(8, Math.min(64, Math.floor(Math.sqrt(cacheBytes / 4))));

    const tileIndices = [];
    const tileSizes = [];
    for (let i = 0; i < Math.min(2, loops.length); i++) {
      const ext = loops[i].extent.type === 'IntImmNode' ? loops[i].extent.value : null;
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

  matches(primFunc, blockName, target) {
    if (!target.isGPU()) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (!isMatmulShape(info)) return false;
    const smemBytes = target.sharedMemoryBytes || 49152;
    const bytesPerTile = 4 * 2;
    const tileDim = Math.max(16, Math.min(128, Math.floor(Math.sqrt(smemBytes / bytesPerTile))));
    const maxExtent = info.loops.reduce((m, l) => {
      const e = l.extent && l.extent.type === 'IntImmNode' ? l.extent.value : 0;
      return e > m ? e : m;
    }, 0);
    return maxExtent >= tileDim;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const info = classifyBlock(schedule.func, blockName);
    const spatial = loops.filter(l => !info || !isReductionLoop(l, info));
    if (spatial.length === 0) return;

    let fused = spatial[0];
    for (let i = 1; i < spatial.length; i++) {
      const current = schedule.getLoops(blockName);
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

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.WASM) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (info.hasReduction || info.loopCount < 1) return false;
    if (info.loops.length > 0 && hasMultipleBlocks(info.loops[0])) return false;
    return true;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
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

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.WASM) return false;
    const hasParallel = target.numCores > 1;
    const hasSimd = target.supportsSimd && target.supportsSimd();
    if (!hasParallel && !hasSimd) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction && info.loopCount >= 2;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
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

  matches() {
    return true;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;
    if (target.isCPU() && loops.length >= 1) {
      schedule.parallelize(loops[0]);
    }
  }
}

export function isReductionLoop(loop, blockInfo) {
  if (!blockInfo.reductionLoopVars) return false;
  return blockInfo.reductionLoopVars.has(loop.loopVar.name);
}

function findDirectChild(parent, child) {
  return parent.body === child;
}

export class SchedulePolicy {
  constructor(target, rules = null, trace = null) {
    this.target = target;
    this.rules = rules || SchedulePolicy.defaultRules();
    this.trace = trace;
  }

  static defaultRules() {
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

  selectRule(primFunc, blockName) {
    for (const rule of this.rules) {
      if (rule.matches(primFunc, blockName, this.target)) {
        return rule;
      }
    }
    return null;
  }

  applyToBlock(schedule, blockName) {
    if (this.target.isGPU() && blockHasNonConstExtent(schedule.func, blockName)) {
      this._explain(blockName, 'none', 'block has dynamic loop extents; runs sequentially (no dynamic grid)');
      return null;
    }
    const rule = this.selectRule(schedule.func, blockName);
    if (rule) {
      rule.apply(schedule, blockName, this.target);
      invalidateClassifyCache(schedule.func);
      this._explain(blockName, rule.name, `matched rule '${rule.name}' for ${this.target.name}`);
      return rule.name;
    }
    this._explain(blockName, 'none', 'no schedule rule matched; runs sequentially');
    return null;
  }

  _explain(blockName, decision, reason) {
    if (this.trace && this.trace.explainsEnabled) {
      this.trace.explain('schedule', blockName, decision, reason, { target: this.target.name });
    }
  }

  applyToAllBlocks(schedule) {
    invalidateClassifyCache(schedule.func);
    const blocks = collectAllBlockNames(schedule.func.body);
    const seen = new Set();
    const applied = new Map();
    for (const name of blocks) {
      if (seen.has(name)) continue;
      seen.add(name);
      const ruleName = this.applyToBlock(schedule, name);
      if (ruleName) applied.set(name, ruleName);
    }
    return applied;
  }
}

function collectAllBlockNames(root) {
  return irCollect(root, (n) => n.type === 'BlockNode', { kinds: 'stmt' }).map((n) => n.name);
}
