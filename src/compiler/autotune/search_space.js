
import { TargetKind } from '../../backend/target.js';

import { classifyBlock, isReductionLoop, primFuncHasReduction } from '../schedule/rules.js';
import { matmulTileDims, enumerateRegisterBlockConfigs, createMatmulRegisterBlockGPUSketch } from './gpu_matmul_sketch.js';

export class SearchVariable {
  constructor(name, candidates) {
    this.name = name;
    this.candidates = candidates;
  }

  sample(rng) {
    return this.candidates[rng(this.candidates.length)];
  }
}

export class ScheduleSketch {
  constructor(name, variables, apply) {
    this.name = name;
    this.variables = variables;
    this._apply = apply;
  }

  instantiate(params) {
    return (schedule, blockName, target) => {
      this._apply(schedule, blockName, target, params);
    };
  }

  sampleParams(rng) {
    const params = {};
    for (const v of this.variables) {
      params[v.name] = v.sample(rng);
    }
    return params;
  }
}

const TILE_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const BLOCK_SIZE_CANDIDATES = [32, 64, 128, 256, 512, 1024];
const UNROLL_CANDIDATES = [1, 2, 4, 8, 16];
const VECTOR_CANDIDATES = [1, 2, 4, 8, 16];

function gpuThreadCap(target) {
  return Math.min((target && target.maxThreadsPerBlock) || 256, 256);
}
const REDUCE_TILE_CANDIDATES = [1, 2, 4, 8, 16, 32];

export function createElementwiseCPUSketch() {
  return new ScheduleSketch('elementwise_cpu', [
    new SearchVariable('tile_size', TILE_CANDIDATES),
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
  return new ScheduleSketch('reduction_cpu', [
    new SearchVariable('tile_size', TILE_CANDIDATES)
  ], (schedule, blockName, target, params) => {
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

export function createMatmulCPUSketch() {
  return new ScheduleSketch('matmul_cpu', [
    new SearchVariable('tile_m', TILE_CANDIDATES),
    new SearchVariable('tile_n', TILE_CANDIDATES),
    new SearchVariable('tile_k', [1, 2, 4, 8, 16, 32])
  ], (schedule, blockName, target, params) => {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const mLoop = loops[0];
    const nLoop = loops[1];

    const mExtent = mLoop.extent.type === 'IntImmNode' ? mLoop.extent.value : null;
    const nExtent = nLoop.extent.type === 'IntImmNode' ? nLoop.extent.value : null;

    if (mExtent && mExtent >= params.tile_m) {
      const [mo, mi] = schedule.split(mLoop, params.tile_m);
      schedule.parallelize(mo);
    }

    const updatedLoops = schedule.getLoops(blockName);
    const currentN = updatedLoops.find(l => l.loopVar.name === nLoop.loopVar.name);
    if (currentN && nExtent && nExtent >= params.tile_n) {
      schedule.split(currentN, params.tile_n);
    }
  });
}

export function createMatmulGPUSketch() {
  return new ScheduleSketch('matmul_gpu', [
    new SearchVariable('block_tile_m', [32, 64, 128]),
    new SearchVariable('block_tile_n', [32, 64, 128]),
    new SearchVariable('thread_tile_m', [4, 8, 16]),
    new SearchVariable('thread_tile_n', [4, 8, 16])
  ], (schedule, blockName, target, params) => {
    const loops = schedule.getLoops(blockName);
    if (loops.length < 3) return;

    const mLoop = loops[0];
    const nLoop = loops[1];
    const mExtent = mLoop.extent.type === 'IntImmNode' ? mLoop.extent.value : null;
    const nExtent = nLoop.extent.type === 'IntImmNode' ? nLoop.extent.value : null;

    const cap = gpuThreadCap(target);
    const threadsOf = (bt, tt) => Math.max(1, Math.ceil(bt / tt));
    let ttm = Math.max(1, params.thread_tile_m);
    let ttn = Math.max(1, params.thread_tile_n);
    while (threadsOf(params.block_tile_m, ttm) * threadsOf(params.block_tile_n, ttn) > cap) {
      if (threadsOf(params.block_tile_m, ttm) >= threadsOf(params.block_tile_n, ttn)) ttm *= 2;
      else ttn *= 2;
      if (ttm >= params.block_tile_m && ttn >= params.block_tile_n) break;
    }

    if (mExtent && mExtent >= params.block_tile_m) {
      const [mBlock, mRem] = schedule.split(mLoop, params.block_tile_m);
      schedule.bindThread(mBlock, 'blockIdx.y');
      const updated = schedule.getLoops(blockName);
      const cur = updated.find(l => l.loopVar.name === mRem.loopVar.name);
      if (cur && ttm <= params.block_tile_m) {
        const [mto] = schedule.split(cur, ttm);
        schedule.bindThread(mto, 'threadIdx.y');
      }
    }

    const updated2 = schedule.getLoops(blockName);
    const currentN = updated2.find(l => l.loopVar.name === nLoop.loopVar.name);
    if (currentN && nExtent && nExtent >= params.block_tile_n) {
      const [nBlock, nRem] = schedule.split(currentN, params.block_tile_n);
      schedule.bindThread(nBlock, 'blockIdx.x');
      const updated3 = schedule.getLoops(blockName);
      const curN = updated3.find(l => l.loopVar.name === nRem.loopVar.name);
      if (curN && ttn <= params.block_tile_n) {
        const [nto] = schedule.split(curN, ttn);
        schedule.bindThread(nto, 'threadIdx.x');
      }
    }
  });
}

export function createMultiLevelTileCPUSketch() {
  return new ScheduleSketch('mlt_cpu', [
    new SearchVariable('tile_s0', TILE_CANDIDATES),
    new SearchVariable('tile_s1', TILE_CANDIDATES),
    new SearchVariable('tile_k', REDUCE_TILE_CANDIDATES)
  ], (schedule, blockName, target, params) => {
    const info = classifyBlock(schedule.func, blockName);
    const loops = schedule.getLoops(blockName);
    if (loops.length === 0) return;

    const isRed = (l) => !!(info && isReductionLoop(l, info));
    const spatial = loops.filter(l => !isRed(l));
    const reduction = loops.filter(l => isRed(l));
    if (spatial.length === 0) return;

    const staticExtent = (l) => (l.extent.type === 'IntImmNode' ? l.extent.value : null);
    const findLoop = (name) => schedule.getLoops(blockName).find(l => l.loopVar.name === name);

    const s0e = staticExtent(spatial[0]);
    let outer = spatial[0];
    if (s0e !== null && params.tile_s0 > 1 && s0e >= params.tile_s0) {
      const [s0o] = schedule.split(spatial[0], params.tile_s0);
      outer = s0o;
    }
    schedule.parallelize(outer);

    if (spatial.length >= 2) {
      const s1 = findLoop(spatial[1].loopVar.name);
      const s1e = s1 ? staticExtent(s1) : null;
      if (s1 && s1e !== null && params.tile_s1 > 1 && s1e >= params.tile_s1) {
        const [, s1i] = schedule.split(s1, params.tile_s1);
        schedule.vectorize(s1i);
      }
    }

    if (reduction.length >= 1) {
      const k = findLoop(reduction[0].loopVar.name);
      const ke = k ? staticExtent(k) : null;
      if (k && ke !== null && params.tile_k > 1 && ke >= params.tile_k) {
        schedule.split(k, params.tile_k);
      }
    }
  });
}

function analyzeBlockStructure(primFunc, blockName) {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return { spatial: 0, reduction: 0, reads: 0, hasReduction: false };
  let spatial = 0;
  let reduction = 0;
  for (const l of info.loops) {
    if (info.reductionLoopVars.has(l.loopVar.name)) reduction++;
    else spatial++;
  }
  return { spatial, reduction, reads: info.readBuffers.length, hasReduction: info.hasReduction };
}

const CPU_SKETCH_RULES = [
  { match: (s) => s.hasReduction && s.spatial >= 1 && s.reads >= 2,
    derive: () => [createMultiLevelTileCPUSketch(), createReductionCPUSketch()] },
  { match: (s) => s.hasReduction,
    derive: () => [createReductionCPUSketch()] },
  { match: () => true,
    derive: () => [createElementwiseCPUSketch()] }
];

const GPU_SKETCH_RULES = [
  { match: (s) => s.hasReduction && s.spatial === 2 && s.reads >= 2,
    derive: () => [createMatmulGPUSketch(), createReductionGPUSketch()] },
  { match: (s) => s.hasReduction,
    derive: () => [createReductionGPUSketch()] },
  { match: () => true,
    derive: () => [createElementwiseGPUSketch()] }
];

function collectAllBlockNames(root) {
  const names = [];
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) continue;
    if (node.type === 'BlockNode') names.push(node.name);
    if (node.body) stack.push(node.body);
    if (node.stmts) for (const s of node.stmts) stack.push(s);
    if (node.thenBody) stack.push(node.thenBody);
    if (node.elseBody) stack.push(node.elseBody);
    if (node.initBody) stack.push(node.initBody);
  }
  return names;
}

export function analyzePureMatmul(primFunc, target) {
  const names = collectAllBlockNames(primFunc.body);
  let reductionBlock = null;
  for (const name of names) {
    const s = analyzeBlockStructure(primFunc, name);
    if (s.hasReduction && s.spatial === 2 && s.reads >= 2) {
      if (reductionBlock) return null;
      reductionBlock = name;
    }
  }
  if (!reductionBlock) return null;
  const dims = matmulTileDims(primFunc, reductionBlock);
  if (!dims) return null;
  for (const name of names) {
    if (name === reductionBlock) continue;
    const info = classifyBlock(primFunc, name);
    if (!info) return null;
    if (info.hasReduction) return null;
    if (info.readBuffers.length > 0) return null;
    for (const w of info.writeBuffers) if (w !== dims.C.name) return null;
  }
  return { reductionBlock, dims };
}

let _matmulSketchCache = null;
let _matmulSketchOwner = null;

function richMatmulSketches(primFunc, blockName, target) {
  const plan = analyzePureMatmul(primFunc, target);
  if (!plan) return null;
  if (_matmulSketchOwner !== primFunc) {
    const configs = enumerateRegisterBlockConfigs(target, plan.dims);
    _matmulSketchCache = configs.length > 0 ? createMatmulRegisterBlockGPUSketch(configs) : null;
    _matmulSketchOwner = primFunc;
  }
  if (!_matmulSketchCache) return null;
  if (blockName === plan.reductionBlock) return [_matmulSketchCache];
  return [];
}

export function getSketchesForBlock(primFunc, blockName, target, blockMap, opts = {}) {
  if (opts.richGpu && target.isGPU()) {
    const rich = richMatmulSketches(primFunc, blockName, target);
    if (rich !== null) return rich;
  }

  const rules = target.kind === TargetKind.CPU ? CPU_SKETCH_RULES
    : (target.isGPU() ? GPU_SKETCH_RULES : null);
  if (!rules) return [];

  const struct = analyzeBlockStructure(primFunc, blockName);
  for (const rule of rules) {
    if (rule.match(struct)) return rule.derive();
  }
  return [];
}
