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
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';
import type { ScheduleSketch } from './sketch.js';
import type { BlockStructure } from './block_analysis.js';
import type { BlockDAG } from './block_dag.js';

export type SketchRule = {
  matches(struct: BlockStructure, target: ScheduleTarget): boolean;
  derive(primFunc: PrimFunc, blockName: string, target: ScheduleTarget, dag?: BlockDAG): ScheduleSketch[];
  priority?: number;
};
export type DeriveOpts = Readonly<{ richGpu?: boolean; dag?: BlockDAG }>;

function reductionSketch(target: ScheduleTarget): ScheduleSketch {
  return target.isGPU() ? createReductionGPUSketch() : createReductionCPUSketch();
}

function elementwiseSketch(target: ScheduleTarget): ScheduleSketch {
  return target.isGPU() ? createElementwiseGPUSketch() : createElementwiseCPUSketch();
}

function deriveMultiLevel(primFunc: PrimFunc, blockName: string, target: ScheduleTarget, dag?: BlockDAG): ScheduleSketch[] {
  const info = classifyBlock(primFunc, blockName);
  if (!info) return [reductionSketch(target)];
  const sketches: ScheduleSketch[] = [];
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

const _sketchRules: Required<SketchRule>[] = [];

export function registerSketchRule(rule: SketchRule, { priority = 100 }: { priority?: number } = {}): void {
  if (_sketchRules.some(r => r.derive === rule.derive && r.matches === rule.matches)) return;
  _sketchRules.push({ matches: rule.matches, derive: rule.derive, priority });
  _sketchRules.sort((a, b) => a.priority - b.priority);
}

registerSketchRule({
  matches: (s: BlockStructure) => s.hasReduction && s.spatial >= 1 && s.reads >= 2,
  derive: deriveMultiLevel,
}, { priority: 10 });
registerSketchRule({
  matches: (s: BlockStructure) => s.hasReduction,
  derive: (primFunc: PrimFunc, blockName: string, target: ScheduleTarget) => [reductionSketch(target)],
}, { priority: 20 });
registerSketchRule({
  matches: () => true,
  derive: (primFunc: PrimFunc, blockName: string, target: ScheduleTarget) => [elementwiseSketch(target)],
}, { priority: 30 });

export function deriveSketches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget, opts: DeriveOpts = {}): ScheduleSketch[] {
  if (opts.richGpu && target.isGPU()) {
    const rich = richMatmulSketches(primFunc, blockName, target);
    if (rich !== null) return rich;
  }
  if (target.kind !== TargetKind.CPU && !target.isGPU()) return [];

  const struct = analyzeBlockStructure(primFunc, blockName);
  const dag = opts.dag || buildBlockDAG(primFunc);
  for (const rule of _sketchRules) {
    if (rule.matches(struct, target)) return rule.derive(primFunc, blockName, target, dag);
  }
  return [];
}
