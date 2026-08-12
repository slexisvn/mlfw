import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

export type TileAxisKind = 'S' | 'R';
export type TileAxis = readonly [kind: TileAxisKind, level: number];
export type TileStructure = {
  name: string;
  order: readonly TileAxis[];
  roles: Readonly<Record<string, string>>;
};

const CPU_TILING: TileStructure = {
  name: 'mlt_cpu',
  order: [['S', 0], ['S', 1], ['S', 2], ['S', 3], ['R', 0]],
  roles: { S0: 'parallelize', S3: 'vectorize' }
};

const GPU_TILING: TileStructure = {
  name: 'mlt_gpu',
  order: [['S', 0], ['S', 1], ['S', 2], ['R', 0]],
  roles: { S0: 'blockIdx', S1: 'threadIdx', S2: 'unroll' }
};

export const CPU_TILING_SSRSRS: TileStructure = {
  name: 'ssrsrs_cpu',
  order: [['S', 0], ['S', 1], ['R', 0], ['S', 2], ['R', 1], ['S', 3]],
  roles: { S0: 'parallelize', S3: 'vectorize', R1: 'unroll' }
};

export function getTileStructure(target: ScheduleTarget): TileStructure {
  return target.isGPU() ? GPU_TILING : CPU_TILING;
}

export function levelCounts(structure: TileStructure): { spatialLevels: number; reductionLevels: number } {
  let spatialLevels = 0;
  let reductionLevels = 0;
  for (const [kind, level] of structure.order) {
    if (kind === 'S') spatialLevels = Math.max(spatialLevels, level + 1);
    else reductionLevels = Math.max(reductionLevels, level + 1);
  }
  return { spatialLevels, reductionLevels };
}
