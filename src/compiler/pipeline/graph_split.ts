import { splitGraphForCublas, splitGraphForNative, hasDependentBoundaries } from '../passes/partition/cublas_split.js';
import { splitGraphForScan } from '../passes/partition/scan_split.js';
import type { GraphModule } from '../ir/graph/module.js';
import type { CompileTarget, CompilerConfig } from './pipeline_types.js';

export type GraphSplitContext = {
  config: CompilerConfig;
  target: CompileTarget & { maxThreadsPerBlock?: number };
  cudaAttention?: boolean;
  cudaMatmulChain?: boolean;
  cudaConvChain?: boolean;
  isWebGPU?: boolean;
};

export type GraphSplitStrategy = {
  name: string;
  priority?: number;
  applies(ctx: GraphSplitContext): boolean | undefined;
  run(graphModule: GraphModule, ctx: GraphSplitContext): unknown;
};

const _strategies: GraphSplitStrategy[] = [];

export function registerGraphSplitStrategy(strategy: GraphSplitStrategy): void {
  _strategies.push(strategy);
  _strategies.sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
}

export function selectGraphSplitStrategy(ctx: GraphSplitContext): GraphSplitStrategy | null {
  for (const s of _strategies) {
    if (s.applies(ctx)) return s;
  }
  return null;
}

export function splitGraph(graphModule: GraphModule, ctx: GraphSplitContext): unknown {
  const strategy = selectGraphSplitStrategy(ctx);
  return strategy ? strategy.run(graphModule, ctx) : null;
}

registerGraphSplitStrategy({
  name: 'cublas',
  priority: 10,
  applies: (ctx: GraphSplitContext) => ctx.config.matmulBackend === 'cublas',
  run: (graphModule: GraphModule) => splitGraphForCublas(graphModule),
});

registerGraphSplitStrategy({
  name: 'cuda-attention',
  priority: 15,
  applies: (ctx: GraphSplitContext) => ctx.cudaAttention,
  run: (graphModule: GraphModule) => splitGraphForNative(graphModule, 1),
});

registerGraphSplitStrategy({
  name: 'cuda-native-matmul-chain',
  priority: 20,
  applies: (ctx: GraphSplitContext) => ctx.cudaMatmulChain,
  run: (graphModule: GraphModule) => splitGraphForNative(graphModule),
});

registerGraphSplitStrategy({
  name: 'cuda-native-conv-chain',
  priority: 25,
  applies: (ctx: GraphSplitContext) => ctx.cudaConvChain,
  run: (graphModule: GraphModule) => splitGraphForNative(graphModule),
});

registerGraphSplitStrategy({
  name: 'webgpu',
  priority: 30,
  applies: (ctx: GraphSplitContext) => ctx.isWebGPU,
  run: (graphModule: GraphModule, ctx: GraphSplitContext) => {
    let split: unknown = splitGraphForScan(graphModule, ctx.target);
    if (!split && hasDependentBoundaries(graphModule, ctx.target.maxThreadsPerBlock || 256)) {
      split = splitGraphForScan(graphModule, ctx.target, true);
      if (!split) split = splitGraphForNative(graphModule, 2);
    }
    return split;
  },
});
