import { splitGraphForCublas, splitGraphForNative, hasDependentBoundaries } from '../passes/partition/cublas_split.js';
import { splitGraphForScan } from '../passes/partition/scan_split.js';
import { countLaunchBoundaries } from '../ir/graph/op_traits.js';
import { TargetAttr, targetAttr } from '../support/target_attrs.js';
import { CUBLAS_PROVIDER, isExternalCodegenEnabled } from './external_codegen.js';
import type { GraphSplitThresholds } from '../support/target_attrs.js';
import type { GraphModule } from '../ir/graph/module.js';
import type { CompileTarget, CompilerConfig } from './pipeline_types.js';

export type GraphSplitTarget = CompileTarget & { maxThreadsPerBlock?: number };

export type GraphSplitContext = Readonly<{
  config: CompilerConfig;
  target: GraphSplitTarget;
  module: GraphModule;
  boundaries: ReadonlyMap<string, number>;
}>;

export type GraphSplitStrategy = Readonly<{
  name: string;
  priority?: number;
  applies(ctx: GraphSplitContext): boolean;
  run(module: GraphModule, ctx: GraphSplitContext): unknown;
}>;

const DEFAULT_WORKGROUP_SIZE = 256;

const _strategies: GraphSplitStrategy[] = [];

export function registerGraphSplitStrategy(strategy: GraphSplitStrategy): void {
  _strategies.push(strategy);
  _strategies.sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
}

export function countBoundaryClasses(module: GraphModule): Map<string, number> {
  const totals = new Map<string, number>();
  for (const func of module) {
    for (const [cls, n] of countLaunchBoundaries(func.ops())) totals.set(cls, (totals.get(cls) || 0) + n);
  }
  return totals;
}

function boundaryThreshold(target: GraphSplitTarget, cls: string): number {
  const thresholds = targetAttr<GraphSplitThresholds>(target, TargetAttr.GRAPH_SPLIT);
  const t = thresholds ? thresholds[cls] : undefined;
  return typeof t === 'number' ? t : Infinity;
}

function reachesBoundaryThreshold(ctx: GraphSplitContext, cls: string): boolean {
  return (ctx.boundaries.get(cls) || 0) >= boundaryThreshold(ctx.target, cls);
}

export function selectGraphSplitStrategy(ctx: GraphSplitContext): GraphSplitStrategy | null {
  for (const s of _strategies) {
    if (s.applies(ctx)) return s;
  }
  return null;
}

export function splitGraph(module: GraphModule, config: CompilerConfig, target: GraphSplitTarget): unknown {
  const ctx: GraphSplitContext = { config, target, module, boundaries: countBoundaryClasses(module) };
  const strategy = selectGraphSplitStrategy(ctx);
  return strategy ? strategy.run(module, ctx) : null;
}

registerGraphSplitStrategy({
  name: CUBLAS_PROVIDER,
  priority: 10,
  applies: (ctx) => isExternalCodegenEnabled(CUBLAS_PROVIDER, ctx.config, ctx.target),
  run: (module) => splitGraphForCublas(module),
});

registerGraphSplitStrategy({
  name: 'attention-chain',
  priority: 15,
  applies: (ctx) => reachesBoundaryThreshold(ctx, 'attention'),
  run: (module) => splitGraphForNative(module, 1),
});

registerGraphSplitStrategy({
  name: 'matmul-chain',
  priority: 20,
  applies: (ctx) => reachesBoundaryThreshold(ctx, 'matmul'),
  run: (module) => splitGraphForNative(module),
});

registerGraphSplitStrategy({
  name: 'conv-chain',
  priority: 25,
  applies: (ctx) => reachesBoundaryThreshold(ctx, 'conv'),
  run: (module) => splitGraphForNative(module),
});

registerGraphSplitStrategy({
  name: 'webgpu-scan',
  priority: 30,
  applies: (ctx) => typeof ctx.target.isWebGPU === 'function' && ctx.target.isWebGPU(),
  run: (module, ctx) => {
    const split = splitGraphForScan(module, ctx.target);
    if (split) return split;
    if (!hasDependentBoundaries(module, ctx.target.maxThreadsPerBlock || DEFAULT_WORKGROUP_SIZE)) return null;
    return splitGraphForScan(module, ctx.target, true) || splitGraphForNative(module, 2);
  },
});
