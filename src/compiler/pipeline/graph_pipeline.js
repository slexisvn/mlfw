import { FixedPointGroup } from '../passes/pass_manager.js';
import { CanonicalizePass } from '../passes/canonicalize/canonicalize.js';
import { AlgebraicSimplificationPass } from '../passes/simplify/algebraic.js';
import { ConstantFoldPass } from '../passes/simplify/constant_fold.js';
import { CSEPass } from '../passes/simplify/cse.js';
import { DCEPass } from '../passes/simplify/dce.js';
import { FusionPass } from '../passes/fusion/fusion_pass.js';
import { EpilogueFusionPass } from '../passes/fusion/epilogue_fusion.js';
import { FusionMergerPass } from '../passes/fusion/fusion_merger.js';
import { MultiOutputFusionPass } from '../passes/fusion/multi_output_fusion.js';
import { DominatorFusionPass } from '../passes/fusion/dominator_fusion.js';
import { PriorityFusionPass } from '../passes/fusion/priority_fusion.js';
import { LayoutTransformPass } from '../passes/layout/layout_transform.js';
import { QuantizationPass } from '../passes/quantization/quantization_pass.js';
import { DecompositionPass } from '../passes/decompose/decomposition_pass.js';
import { RematerializationPass } from '../passes/memory/rematerialization.js';
import { CublasRewritePass } from '../passes/rewrite/cublas_rewrite.js';
import { graphPassesForPhase } from './graph_pass_registry.js';

const DEFAULT_LAUNCH_OVERHEAD_US = 5;

export function buildGraphPipeline(config, target, { cudaMatmulChain = false, context = null } = {}) {
  const passesForPhase = context
    ? (phase) => context.passesForPhase(phase, config, target)
    : (phase) => graphPassesForPhase(phase, config, target);
  const passes = [];

  for (const p of passesForPhase('pre')) passes.push(p);

  passes.push(new DecompositionPass(target));
  passes.push(new FixedPointGroup('canonicalize', [
    new CanonicalizePass(),
    new AlgebraicSimplificationPass({ fastMath: config.optimization.fastMath }),
    new ConstantFoldPass(),
    new CSEPass(),
    new DCEPass(),
  ], config.optimization.maxSimplifyIterations));

  if (config.optimization.layout && target) {
    passes.push(new LayoutTransformPass({ target }));
    passes.push(new DCEPass());
  }

  if (config.quantization.enabled) {
    passes.push(new QuantizationPass({ ...config.quantization, target }));
    passes.push(new CanonicalizePass());
    passes.push(new DCEPass());
  }

  const shouldEpilogueFuse = config.matmulBackend !== 'cublas'
    && (config.fusion.epilogue !== undefined
      ? config.fusion.epilogue
      : (target && target.enableEpilogueFusion));

  if (shouldEpilogueFuse) {
    passes.push(new EpilogueFusionPass({ target }));
    passes.push(new DCEPass());
  }

  if (config.fusion.enabled) {
    const fCfg = config.fusion;
    const launchOverheadUs = fCfg.launchOverheadUs ?? DEFAULT_LAUNCH_OVERHEAD_US;
    if (fCfg.strategy === 'dominator') {
      passes.push(new DominatorFusionPass({ target, ...fCfg }));
    } else if (fCfg.strategy === 'priority') {
      passes.push(new PriorityFusionPass({ target, cost: { launchOverheadUs }, ...fCfg }));
      passes.push(new MultiOutputFusionPass({ maxFusionSize: target?.maxFusionSize, ...fCfg }));
    } else {
      passes.push(new FusionPass({ target, cost: { launchOverheadUs }, ...fCfg }));
      passes.push(new FusionMergerPass({ maxFusionSize: target?.maxFusionSize, ...fCfg }));
      passes.push(new MultiOutputFusionPass({ maxFusionSize: target?.maxFusionSize, ...fCfg }));
    }
    passes.push(new DCEPass());
  }

  if (config.matmulBackend === 'cublas') {
    passes.push(new CublasRewritePass());
  }

  if (config.optimization.rematerialization) {
    const rcfg = { ...config.optimization.rematConfig };
    if (rcfg.memoryBudget === undefined && target && target.memoryBudgetBytes > 0) {
      rcfg.memoryBudget = target.memoryBudgetBytes;
    }
    passes.push(new RematerializationPass(rcfg));
  }

  for (const p of passesForPhase('post')) passes.push(p);

  return passes;
}
