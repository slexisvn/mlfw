const CPU_TILING = {
  name: 'mlt_cpu',
  order: [['S', 0], ['S', 1], ['S', 2], ['S', 3], ['R', 0]],
  roles: { S0: 'parallelize', S3: 'vectorize' }
};

const GPU_TILING = {
  name: 'mlt_gpu',
  order: [['S', 0], ['S', 1], ['S', 2], ['R', 0]],
  roles: { S0: 'blockIdx', S1: 'threadIdx', S2: 'unroll' }
};

export const CPU_TILING_SSRSRS = {
  name: 'ssrsrs_cpu',
  order: [['S', 0], ['S', 1], ['R', 0], ['S', 2], ['R', 1], ['S', 3]],
  roles: { S0: 'parallelize', S3: 'vectorize', R1: 'unroll' }
};

export function getTileStructure(target) {
  return target.isGPU() ? GPU_TILING : CPU_TILING;
}

export function levelCounts(structure) {
  let spatialLevels = 0;
  let reductionLevels = 0;
  for (const [kind, level] of structure.order) {
    if (kind === 'S') spatialLevels = Math.max(spatialLevels, level + 1);
    else reductionLevels = Math.max(reductionLevels, level + 1);
  }
  return { spatialLevels, reductionLevels };
}
