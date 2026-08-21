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
import { CalibrationPass } from '../passes/quantization/calibration_pass.js';
import { DecompositionPass } from '../passes/decompose/decomposition_pass.js';
import { RematerializationPass } from '../passes/memory/rematerialization.js';
import { CallInlinerPass } from '../passes/inline/call_inliner.js';
import { activeExternalCodegenProviders } from './external_codegen.js';
import { graphPassesForPhase } from './graph_pass_registry.js';
import type { CompileFn } from '../analysis/calibrate_exec.js';
import type { CompilerContext } from './compiler_context.js';
import type { CompilerConfig, CompileTarget, FusionAwareTarget, GraphPass, PassPhase } from './pipeline_types.js';

export type GraphPipelineOpts = Readonly<{ context?: CompilerContext | null; compileFn?: CompileFn | null }>;

type GraphPipelineTarget = FusionAwareTarget;

const DEFAULT_LAUNCH_OVERHEAD_US = 5;

export function buildGraphPipeline(config: CompilerConfig, target: GraphPipelineTarget | null, { context = null, compileFn = null }: GraphPipelineOpts = {}): GraphPass[] {
  const passesForPhase = context
    ? (phase: PassPhase) => context.passesForPhase(phase, config, target as CompileTarget)
    : (phase: PassPhase) => graphPassesForPhase(phase, config, target as CompileTarget);
  const passes: GraphPass[] = [];

  for (const p of passesForPhase('pre')) passes.push(p);

  passes.push(new CallInlinerPass());
  passes.push(new DecompositionPass(target as unknown as null));
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
    const quantize = new QuantizationPass({ ...config.quantization, target });
    const batches = config.quantization.calibrationData as readonly unknown[] | undefined;
    if (batches && !quantize.config.calibration && target) {
      passes.push(new CalibrationPass({
        quantConfig: quantize.config,
        batches,
        mode: config.quantization.calibrationMode as string | undefined,
        target,
        compileFn: compileFn || undefined,
      }));
    }
    passes.push(quantize);
    passes.push(new CanonicalizePass());
    passes.push(new DCEPass());
  }

  const providers = activeExternalCodegenProviders(config, target as CompileTarget);
  const shouldEpilogueFuse = config.fusion.enabled && !providers.some(p => p.suppressesEpilogueFusion)
    && target && target.enableEpilogueFusion;

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

  for (const provider of providers) {
    if (!provider.graphPasses) continue;
    for (const p of provider.graphPasses(config, target as CompileTarget)) passes.push(p);
  }

  if (config.optimization.rematerialization) {
    const rcfg = { ...(config.optimization.rematConfig as Record<string, unknown>) } as Record<string, unknown>;
    if (rcfg.memoryBudget === undefined && target && (target.memoryBudgetBytes ?? 0) > 0) {
      rcfg.memoryBudget = target.memoryBudgetBytes;
    }
    passes.push(new RematerializationPass(rcfg));
  }

  for (const p of passesForPhase('post')) passes.push(p);

  return passes;
}
