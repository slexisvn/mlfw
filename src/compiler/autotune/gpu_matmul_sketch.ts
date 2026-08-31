import { ScheduleSketch, SearchVariable } from './sketch.js';
import {
  matmulTileDims, buildRegisterBlockedMatmul, enumerateRegisterBlockConfigs, analyzePureMatmul,
} from '../schedule/matmul_tiling.js';
import { FuncAttr } from '../ir/func_attrs.js';
import type { RegisterBlockConfig } from '../schedule/matmul_tiling.js';
import type { SketchParams } from './sketch.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

function createMatmulRegisterBlockGPUSketch(configs: readonly RegisterBlockConfig[]): ScheduleSketch {
  const idxVar = new SearchVariable('config_index', configs.map((_, i) => i));
  const sketch = new ScheduleSketch('matmul_register_block_gpu', [idxVar],
    (schedule, blockName, target, params) => {
      const bufs = matmulTileDims(schedule.func, blockName);
      if (!bufs) return;
      const cfg = configs[params.config_index as number];
      if (!cfg) return;
      const body = buildRegisterBlockedMatmul(bufs, cfg);
      schedule.func.body = body;
      if (schedule.func._setChild) schedule.func._setChild('body', body);
      schedule.func.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
    });
  const enriched = sketch as ScheduleSketch & { configs: readonly RegisterBlockConfig[]; enumerate: () => SketchParams[] };
  enriched.configs = configs;
  enriched.enumerate = () => configs.map((_, i) => ({ config_index: i }));
  return enriched;
}

const matmulSketchCache = new WeakMap();

export function richMatmulSketches(primFunc: PrimFunc, blockName: string, target: ScheduleTarget): ScheduleSketch[] | null {
  const plan = analyzePureMatmul(primFunc);
  if (!plan) return null;
  let sketch = matmulSketchCache.get(primFunc);
  if (sketch === undefined) {
    const configs = enumerateRegisterBlockConfigs(target, plan.dims);
    sketch = configs.length > 0 ? createMatmulRegisterBlockGPUSketch(configs) : null;
    matmulSketchCache.set(primFunc, sketch);
  }
  if (!sketch) return null;
  if (blockName === plan.reductionBlock) return [sketch];
  return [];
}
