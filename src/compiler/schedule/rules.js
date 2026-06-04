import { ForKind } from '../ir/tensor/nodes.js';
import { TargetKind } from '../backend/target.js';

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

function classifyBlock(primFunc, blockName) {
  const blocks = new Map();
  collectBlockInfo(primFunc.body, blocks, []);
  return blocks.get(blockName) || null;
}

function collectBlockInfo(node, result, loopStack) {
  if (!node) return;
  if (node.type === 'ForNode') {
    loopStack.push(node);
    collectBlockInfo(node.body, result, loopStack);
    loopStack.pop();
    return;
  }
  if (node.type === 'BlockNode') {
    const hasReduction = node.initBody !== null;
    const readBuffers = node.reads.map(r => r.buffer.name);
    const writeBuffers = node.writes.map(r => r.buffer.name);
    result.set(node.name, {
      loopCount: loopStack.length,
      hasReduction,
      readBuffers,
      writeBuffers,
      loops: [...loopStack]
    });
    collectBlockInfo(node.body, result, loopStack);
    return;
  }
  if (node.type === 'SeqNode') {
    for (const s of node.stmts) collectBlockInfo(s, result, loopStack);
    return;
  }
  if (node.type === 'IfThenElseNode') {
    collectBlockInfo(node.thenBody, result, loopStack);
    if (node.elseBody) collectBlockInfo(node.elseBody, result, loopStack);
  }
}

export class ElementwiseCPURule extends ScheduleRule {
  constructor() {
    super('elementwise_cpu');
  }

  matches(primFunc, blockName, target) {
    if (target.kind !== TargetKind.CPU) return false;
    const info = classifyBlock(primFunc, blockName);
    if (!info) return false;
    return !info.hasReduction && info.loopCount >= 1;
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

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const tileM = 32;
    const tileN = 32;
    const tileK = 8;

    const mLoop = loops[0];
    const nLoop = loops[1];
    const kLoop = loops[2];

    const mExtent = mLoop.extent.type === 'IntImmNode' ? mLoop.extent.value : null;
    const nExtent = nLoop.extent.type === 'IntImmNode' ? nLoop.extent.value : null;

    if (mExtent && mExtent >= tileM) {
      const [mo, mi] = schedule.split(mLoop, tileM);
      schedule.parallelize(mo);
    }

    const updatedLoops = schedule.getLoops(blockName);
    const currentN = updatedLoops.find(l => l.loopVar.name === nLoop.loopVar.name);
    if (currentN && nExtent && nExtent >= tileN) {
      schedule.split(currentN, tileN);
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
    return info.hasReduction && blockName.includes('matmul');
  }

  apply(schedule, blockName, target) {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const mLoop = loops[0];
    const nLoop = loops[1];

    const blockTileM = 64;
    const blockTileN = 64;
    const threadTileM = 8;
    const threadTileN = 8;

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
      return rule.name;
    }
    return null;
  }

  applyToAllBlocks(schedule) {
    const blocks = collectAllBlockNames(schedule.func.body);
    const applied = new Map();
    for (const name of blocks) {
      const ruleName = this.applyToBlock(schedule, name);
      if (ruleName) applied.set(name, ruleName);
    }
    return applied;
  }
}

function collectAllBlockNames(node, result = []) {
  if (!node) return result;
  if (node.type === 'BlockNode') {
    result.push(node.name);
  }
  if (node.body) collectAllBlockNames(node.body, result);
  if (node.stmts) {
    for (const s of node.stmts) collectAllBlockNames(s, result);
  }
  if (node.thenBody) collectAllBlockNames(node.thenBody, result);
  if (node.elseBody) collectAllBlockNames(node.elseBody, result);
  if (node.initBody) collectAllBlockNames(node.initBody, result);
  return result;
}
