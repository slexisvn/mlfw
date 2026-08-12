import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';
import type { ScheduleSketch } from './sketch.js';

import { deriveSketches } from './derivation.js';

export function getSketchesForBlock(primFunc: PrimFunc, blockName: string, target: ScheduleTarget, blockMap?: unknown, opts: Record<string, unknown> = {}): ScheduleSketch[] {
  return deriveSketches(primFunc, blockName, target, opts);
}
