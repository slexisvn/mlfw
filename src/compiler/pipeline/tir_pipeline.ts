import { SchedulePass } from '../passes/schedule/schedule_pass.js';
import { SimplifyPass } from '../passes/simplify/simplify_pass.js';
import { MemoryPlanPass } from '../passes/memory/memory_plan_pass.js';
import { MemorySchedulePass } from '../passes/memory/memory_scheduler.js';
import { LoopPartitionPass } from '../passes/loop_partition/loop_partition.js';
import { AccumulatorDetectionPass } from '../passes/lowering/accumulator_pass.js';
import { AutoTensorizePass } from '../passes/schedule/tensorize_pass.js';
import { InlineReindexPass } from '../passes/schedule/inline_reindex_pass.js';
import { tirPassesForPhase } from './tir_pass_registry.js';
import type { CompilerConfig, TirPass } from './pipeline_types.js';

export function buildTirPipeline(config: CompilerConfig): TirPass[] {
  const passes: TirPass[] = [];

  for (const p of tirPassesForPhase('pre', config)) passes.push(p);

  passes.push(new InlineReindexPass(config));

  if (config.optimization.tensorize) passes.push(new AutoTensorizePass(config));

  passes.push(new SchedulePass(config));

  if (config.optimization.loopPartition) passes.push(new LoopPartitionPass());

  passes.push(new SimplifyPass());

  if (config.memory.scheduleForPeak) passes.push(new MemorySchedulePass(config));

  passes.push(new MemoryPlanPass(config));

  if (config.optimization.detectAccumulators) passes.push(new AccumulatorDetectionPass());

  for (const p of tirPassesForPhase('post', config)) passes.push(p);

  return passes;
}
