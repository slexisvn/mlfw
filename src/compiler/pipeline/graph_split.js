import { splitGraphForCublas, splitGraphForNative, hasDependentBoundaries } from '../passes/partition/cublas_split.js';
import { splitGraphForScan } from '../passes/partition/scan_split.js';

const _strategies = [];

export function registerGraphSplitStrategy(strategy) {
  _strategies.push(strategy);
  _strategies.sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
}

export function selectGraphSplitStrategy(ctx) {
  for (const s of _strategies) {
    if (s.applies(ctx)) return s;
  }
  return null;
}

export function splitGraph(graphModule, ctx) {
  const strategy = selectGraphSplitStrategy(ctx);
  return strategy ? strategy.run(graphModule, ctx) : null;
}

registerGraphSplitStrategy({
  name: 'cublas',
  priority: 10,
  applies: (ctx) => ctx.config.matmulBackend === 'cublas',
  run: (graphModule) => splitGraphForCublas(graphModule),
});

registerGraphSplitStrategy({
  name: 'cuda-attention',
  priority: 15,
  applies: (ctx) => ctx.cudaAttention,
  run: (graphModule) => splitGraphForNative(graphModule, 1),
});

registerGraphSplitStrategy({
  name: 'cuda-native-matmul-chain',
  priority: 20,
  applies: (ctx) => ctx.cudaMatmulChain,
  run: (graphModule) => splitGraphForNative(graphModule),
});

registerGraphSplitStrategy({
  name: 'cuda-native-conv-chain',
  priority: 25,
  applies: (ctx) => ctx.cudaConvChain,
  run: (graphModule) => splitGraphForNative(graphModule),
});

registerGraphSplitStrategy({
  name: 'webgpu',
  priority: 30,
  applies: (ctx) => ctx.isWebGPU,
  run: (graphModule, ctx) => {
    let split = splitGraphForScan(graphModule, ctx.target);
    if (!split && hasDependentBoundaries(graphModule, ctx.target.maxThreadsPerBlock || 256)) {
      split = splitGraphForScan(graphModule, ctx.target, true);
      if (!split) split = splitGraphForNative(graphModule, 2);
    }
    return split;
  },
});
