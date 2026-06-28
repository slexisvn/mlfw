import { SchedulePass } from '../passes/schedule/schedule_pass.js';
import { SimplifyPass } from '../passes/simplify/simplify_pass.js';
import { MemoryPlanPass } from '../passes/memory/memory_plan_pass.js';
import { LoopPartitionPass } from '../passes/loop_partition/loop_partition.js';
import { AccumulatorDetectionPass } from '../passes/lowering/accumulator_pass.js';

export function buildTirPipeline(config) {
  const passes = [];

  passes.push(new SchedulePass(config));

  if (config.optimization.loopPartition) passes.push(new LoopPartitionPass());

  passes.push(new SimplifyPass());

  passes.push(new MemoryPlanPass(config));

  if (config.optimization.detectAccumulators) passes.push(new AccumulatorDetectionPass());

  return passes;
}
