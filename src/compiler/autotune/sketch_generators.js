import { classifyBlock, isReductionLoop, primFuncHasReduction } from '../schedule/rules.js';
import { ScheduleSketch, SearchVariable } from './sketch.js';

const BLOCK_SIZE_CANDIDATES = [32, 64, 128, 256, 512, 1024];
const VECTOR_CANDIDATES = [1, 2, 4, 8, 16];

function gpuThreadCap(target) {
  return Math.min((target && target.maxThreadsPerBlock) || 256, 256);
}

function reductionDivisors(K) {
  const out = new Set();
  for (let d = 2; d * d <= K; d++) {
    if (K % d !== 0) continue;
    out.add(d);
    const e = K / d;
    if (e > 1 && e < K) out.add(e);
  }
  return [...out].sort((a, b) => a - b);
}

export function createFusedTilingSketch(consumerName) {
  return new ScheduleSketch('fused', [], (schedule, blockName, target, params) => {
    schedule.fuseConsumer(blockName, consumerName);
    const loops = schedule.getLoops(blockName);
    if (loops.length > 0) schedule.parallelize(loops[0]);
  });
}

export function createRfactorSketch(blockInfo) {
  const reductionLoops = blockInfo.loops.filter(l => blockInfo.reductionLoopVars.has(l.loopVar.name));
  if (reductionLoops.length !== 1) return null;
  const kLoop = reductionLoops[0];
  const K = kLoop.extent && kLoop.extent.type === 'IntImmNode' ? kLoop.extent.value : null;
  if (K === null) return null;
  const factors = reductionDivisors(K);
  if (factors.length === 0) return null;
  const reductionVar = kLoop.loopVar.name;
  return new ScheduleSketch('rfactor', [new SearchVariable('rf_factor', factors)], (schedule, blockName, target, params) => {
    schedule.rfactor(blockName, reductionVar, params.rf_factor);
    const loops = schedule.getLoops(`${blockName}_rf_p`);
    if (loops.length > 0) schedule.parallelize(loops[0]);
  });
}

export function createElementwiseCPUSketch() {
  return new ScheduleSketch('elementwise_cpu', [
    new SearchVariable('vector_width', VECTOR_CANDIDATES)
  ], (schedule, blockName, target, params) => {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    if (loops.length === 1) {
      const extent = loops[0].extent;
      if (extent.type === 'IntImmNode' && extent.value >= params.vector_width * 2) {
        const [outer, inner] = schedule.split(loops[0], params.vector_width);
        schedule.parallelize(outer);
        schedule.vectorize(inner);
      } else {
        schedule.parallelize(loops[0]);
      }
      return;
    }

    schedule.parallelize(loops[0]);
    const innermost = loops[loops.length - 1];
    const extent = innermost.extent;
    if (extent.type === 'IntImmNode' && extent.value >= params.vector_width) {
      const [, vi] = schedule.split(innermost, params.vector_width);
      schedule.vectorize(vi);
    }
  });
}

export function createElementwiseGPUSketch() {
  return new ScheduleSketch('elementwise_gpu', [
    new SearchVariable('block_size', BLOCK_SIZE_CANDIDATES)
  ], (schedule, blockName, target, params) => {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    let fusedLoop = loops[0];
    for (let i = 1; i < loops.length; i++) {
      const currentLoops = schedule.getLoops(blockName);
      const nextLoop = currentLoops.find(l => l.loopVar.name === loops[i].loopVar.name);
      if (nextLoop && fusedLoop.body === nextLoop) {
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
    const blockSize = Math.min(params.block_size, gpuThreadCap(target));
    if (totalElements > blockSize) {
      const [bx, tx] = schedule.split(fusedLoop, blockSize);
      schedule.bindThread(bx, 'blockIdx.x');
      schedule.bindThread(tx, 'threadIdx.x');
    } else {
      schedule.bindThread(fusedLoop, 'threadIdx.x');
    }
  });
}

export function createReductionCPUSketch() {
  return new ScheduleSketch('reduction_cpu', [], (schedule, blockName, target, params) => {
    const loops = schedule.getLoops(blockName);
    if (loops.length > 0) {
      schedule.parallelize(loops[0]);
    }
  });
}

export function createReductionGPUSketch() {
  return new ScheduleSketch('reduction_gpu', [
    new SearchVariable('block_size', BLOCK_SIZE_CANDIDATES)
  ], (schedule, blockName, target, params) => {
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    const info = classifyBlock(schedule.func, blockName);
    const spatialLoops = loops.filter(l => !info || !isReductionLoop(l, info));
    if (spatialLoops.length === 0) return;

    let fused = spatialLoops[0];
    for (let i = 1; i < spatialLoops.length; i++) {
      const current = schedule.getLoops(blockName);
      const next = current.find(l => l.loopVar.name === spatialLoops[i].loopVar.name);
      if (next && fused.body === next) fused = schedule.fuseLoops(fused, next);
    }

    const extent = fused.extent;
    if (extent.type !== 'IntImmNode') {
      schedule.bindThread(fused, 'threadIdx.x');
      return;
    }
    const blockSize = Math.min(params.block_size, gpuThreadCap(target));
    if (extent.value > blockSize) {
      const [bx, tx] = schedule.split(fused, blockSize);
      schedule.bindThread(bx, 'blockIdx.x');
      schedule.bindThread(tx, 'threadIdx.x');
    } else {
      schedule.bindThread(fused, 'threadIdx.x');
    }
  });
}
