import { TargetKind } from '../../backend/target.js';
import { classifyBlock } from '../schedule/rules.js';
import { analyzeBlockStructure } from './block_analysis.js';
import { getTileStructure, CPU_TILING_SSRSRS } from './tile_structure.js';
import { createMultiLevelTilingSketch, createSSRSRSTilingSketch } from './tiling.js';
import {
  createElementwiseCPUSketch, createElementwiseGPUSketch,
  createReductionCPUSketch, createReductionGPUSketch, createRfactorSketch,
  createFusedTilingSketch
} from './sketch_generators.js';
import { richMatmulSketches } from './gpu_matmul_sketch.js';
import { buildBlockDAG, findFusibleConsumer } from './block_dag.js';

function reductionSketch(target) {
  return target.isGPU() ? createReductionGPUSketch() : createReductionCPUSketch();
}

function elementwiseSketch(target) {
  return target.isGPU() ? createElementwiseGPUSketch() : createElementwiseCPUSketch();
}

function deriveMultiLevel(primFunc, blockName, target, dag) {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return [reductionSketch(target)];
  const sketches = [];
  const tiling = createMultiLevelTilingSketch(info, getTileStructure(target));
  if (tiling) sketches.push(tiling);
  if (target.kind === TargetKind.CPU) {
    const ssrsrs = createSSRSRSTilingSketch(info, CPU_TILING_SSRSRS);
    if (ssrsrs) sketches.push(ssrsrs);
    const rf = createRfactorSketch(info);
    if (rf) sketches.push(rf);
    const consumer = dag ? findFusibleConsumer(primFunc, dag, blockName, classifyBlock) : null;
    if (consumer) sketches.push(createFusedTilingSketch(consumer));
  }
  sketches.push(reductionSketch(target));
  return sketches;
}

const _sketchRules = [];

export function registerSketchRule(rule, { priority = 100 } = {}) {
  _sketchRules.push({ matches: rule.matches, derive: rule.derive, priority });
  _sketchRules.sort((a, b) => a.priority - b.priority);
}

registerSketchRule({
  matches: (s) => s.hasReduction && s.spatial >= 1 && s.reads >= 2,
  derive: deriveMultiLevel,
}, { priority: 10 });
registerSketchRule({
  matches: (s) => s.hasReduction,
  derive: (primFunc, blockName, target) => [reductionSketch(target)],
}, { priority: 20 });
registerSketchRule({
  matches: () => true,
  derive: (primFunc, blockName, target) => [elementwiseSketch(target)],
}, { priority: 30 });

export function deriveSketches(primFunc, blockName, target, opts = {}) {
  if (opts.richGpu && target.isGPU()) {
    const rich = richMatmulSketches(primFunc, blockName, target);
    if (rich !== null) return rich;
  }
  if (target.kind !== TargetKind.CPU && !target.isGPU()) return [];

  const struct = analyzeBlockStructure(primFunc, blockName);
  const dag = buildBlockDAG(primFunc);
  for (const rule of _sketchRules) {
    if (rule.matches(struct, target)) return rule.derive(primFunc, blockName, target, dag);
  }
  return [];
}
