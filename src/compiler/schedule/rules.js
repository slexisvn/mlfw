import { ForKind } from '../ir/tensor/nodes.js';
import { TargetKind } from '../../backend/target.js';

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

let _classifyCache = null;
let _classifyCacheOwner = null;

function classifyBlock(primFunc, blockName) {
  if (_classifyCacheOwner !== primFunc || !_classifyCache) {
    _classifyCache = new Map();
    collectBlockInfo(primFunc.body, _classifyCache, []);
    _classifyCacheOwner = primFunc;
  }
  return _classifyCache.get(blockName) || null;
}

function invalidateClassifyCache() {
  _classifyCache = null;
  _classifyCacheOwner = null;
}

function collectBlockInfo(root, result, initialLoopStack) {
  const stack = [{ node: root, loops: [...initialLoopStack] }];
  while (stack.length > 0) {
    const { node, loops } = stack.pop();
    if (!node) continue;
    if (node.type === 'ForNode') {
      stack.push({ node: node.body, loops: [...loops, node] });
    } else if (node.type === 'BlockNode') {
      result.set(node.name, {
        loopCount: loops.length,
        hasReduction: node.initBody !== null,
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
      } else {
        schedule.vectorize(innermost);
      }
    }
  }
}

export class ElementwiseGPURule extends ScheduleRule {
  constructor() {
    super('elementwise_gpu');
  }

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.GPU) return false;
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
    const blockSize = Math.min(target.maxThreadsPerBlock, 256);
    const numBlocks = Math.ceil(totalElements / blockSize);

    if (numBlocks > 1) {
      const [bx, tx] = schedule.split(fusedLoop, blockSize);
      schedule.bindThread(bx, 'blockIdx.x');
      schedule.bindThread(tx, 'threadIdx.x');
    } else {
      schedule.bindThread(fusedLoop, 'threadIdx.x');
    }
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

    for (const loop of loops) {
      const info = classifyBlock(schedule.func, blockName);
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
    if (target.kind !== TargetKind.GPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return info.hasReduction;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    const spatialLoops = [];
    for (const loop of loops) {
      const info = classifyBlock(schedule.func, blockName);
      if (!info || !isReductionLoop(loop, info)) {
        spatialLoops.push(loop);
      }
    }

    if (spatialLoops.length > 0) {
      const blockSize = Math.min(target.maxThreadsPerBlock, 256);
      const extent = spatialLoops[0].extent;
      if (extent.type === 'IntImmNode' && extent.value > blockSize) {
        const [bx, tx] = schedule.split(spatialLoops[0], blockSize);
        schedule.bindThread(bx, 'blockIdx.x');
        schedule.bindThread(tx, 'threadIdx.x');
      } else {
        schedule.bindThread(spatialLoops[0], 'threadIdx.x');
      }
    }
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
    return info.hasReduction && blockName.includes('matmul');
  }

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (!info.hasReduction || !blockName.includes('matmul')) return false;
    if (info.loopCount < 3) return false;
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

    const mLoop = loops[0];
    const nLoop = loops[1];

    const mExtent = mLoop.extent.type === 'IntImmNode' ? mLoop.extent.value : null;
    const nExtent = nLoop.extent.type === 'IntImmNode' ? nLoop.extent.value : null;

    if (mExtent && mExtent >= tileDim) {
      const [mo] = schedule.split(mLoop, tileDim);
      schedule.parallelize(mo);
    }

    const updatedLoops = schedule.getLoops(blockName);
    const currentN = updatedLoops.find(l => l.loopVar.name === nLoop.loopVar.name);
    if (currentN && nExtent && nExtent >= tileDim) {
      schedule.split(currentN, tileDim);
    }
  }
}

export class MatmulTiledGPURule extends ScheduleRule {
  constructor() {
    super('matmul_tiled_gpu');
  }

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.GPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    if (!info.hasReduction || !blockName.includes('matmul')) return false;
    if (info.loopCount < 3) return false;
    const smemBytes = target.sharedMemoryBytes || 49152;
    const tileDim = Math.max(16, Math.min(128, Math.floor(Math.sqrt(smemBytes / 4))));
    const maxExtent = info.loops.reduce((m, l) => {
      const e = l.extent && l.extent.type === 'IntImmNode' ? l.extent.value : 0;
      return e > m ? e : m;
    }, 0);
    return maxExtent >= tileDim;
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const mLoop = loops[0];
    const nLoop = loops[1];

    const smemBytes = target.sharedMemoryBytes || 49152;
    const tcScale = target.supportsTensorCore ? 2 : 1;
    const blockTileDim = Math.max(16, Math.min(128, Math.floor(Math.sqrt(smemBytes / 4)) * tcScale));
    const blockTileM = blockTileDim;
    const blockTileN = blockTileDim;
    const warp = target.warpSize || 32;
    const threadTileM = Math.max(4, Math.min(16, Math.floor(blockTileM / (warp / 4))));
    const threadTileN = Math.max(4, Math.min(16, Math.floor(blockTileN / (warp / 4))));

    const mExtent = mLoop.extent.type === 'IntImmNode' ? mLoop.extent.value : null;
    const nExtent = nLoop.extent.type === 'IntImmNode' ? nLoop.extent.value : null;

    if (mExtent && mExtent >= blockTileM) {
      const [mBlock, mThread] = schedule.split(mLoop, blockTileM);
      schedule.bindThread(mBlock, 'blockIdx.y');

      const updatedLoops = schedule.getLoops(blockName);
      const currentMThread = updatedLoops.find(l => l.loopVar.name === mThread.loopVar.name);
      if (currentMThread) {
        const [mto, mti] = schedule.split(currentMThread, threadTileM);
        schedule.bindThread(mto, 'threadIdx.y');
      }
    }

    const updatedLoops2 = schedule.getLoops(blockName);
    const currentN = updatedLoops2.find(l => l.loopVar.name === nLoop.loopVar.name);
    if (currentN && nExtent && nExtent >= blockTileN) {
      const [nBlock, nThread] = schedule.split(currentN, blockTileN);
      schedule.bindThread(nBlock, 'blockIdx.x');

      const updatedLoops3 = schedule.getLoops(blockName);
      const currentNThread = updatedLoops3.find(l => l.loopVar.name === nThread.loopVar.name);
      if (currentNThread) {
        const [nto, nti] = schedule.split(currentNThread, threadTileN);
        schedule.bindThread(nto, 'threadIdx.x');
      }
    }
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

    const innermost = loops[loops.length - 1];
    const extent = innermost.extent;
    const vectorWidth = target.vectorWidth || 4;

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

function isReductionLoop(loop, blockInfo) {
  const loopIdx = blockInfo.loops.indexOf(loop);
  if (loopIdx < 0) return false;
  const spatialCount = blockInfo.writeBuffers.length > 0
    ? estimateSpatialDims(blockInfo)
    : blockInfo.loopCount - 1;
  return loopIdx >= spatialCount;
}

function estimateSpatialDims(blockInfo) {
  if (blockInfo.hasReduction) {
    return Math.max(1, blockInfo.loopCount - 1);
  }
  return blockInfo.loopCount;
}

function findDirectChild(parent, child) {
  return parent.body === child;
}

export class SchedulePolicy {
  constructor(target, rules = null) {
    this.target = target;
    this.rules = rules || SchedulePolicy.defaultRules();
  }

  static defaultRules() {
    return [
      new MatmulTiledCPURule(),
      new MatmulTiledGPURule(),
      new ReductionCPURule(),
      new ReductionGPURule(),
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
    const rule = this.selectRule(schedule.func, blockName);
    if (rule) {
      rule.apply(schedule, blockName, this.target);
      invalidateClassifyCache();
      return rule.name;
    }
    return null;
  }

  applyToAllBlocks(schedule) {
    invalidateClassifyCache();
    const blocks = collectAllBlockNames(schedule.func.body);
    const applied = new Map();
    for (const name of blocks) {
      const ruleName = this.applyToBlock(schedule, name);
      if (ruleName) applied.set(name, ruleName);
    }
    return applied;
  }
}

function collectAllBlockNames(root) {
  const result = [];
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) continue;
    if (node.type === 'BlockNode') result.push(node.name);
    if (node.body) stack.push(node.body);
    if (node.stmts) for (const s of node.stmts) stack.push(s);
    if (node.thenBody) stack.push(node.thenBody);
    if (node.elseBody) stack.push(node.elseBody);
    if (node.initBody) stack.push(node.initBody);
  }
  return result;
}
