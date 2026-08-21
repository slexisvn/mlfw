// Bundle entry for the book's Part VII labs.
//
// Part VII is about the scheduling language, and `Schedule` is not part of the
// package's public surface: nothing outside the compiler is meant to reshape a
// loop nest by hand. The labs need to, so this file names the internal modules
// they reach for. `docs/part7/_internals.mjs` bundles it with esbuild on the
// first run of any lab; no build step is required of the reader.

export * from '../../src/index.js';

export { Schedule, resetVarCounter } from '../../src/compiler/schedule/schedule.js';
export { ScheduleState } from '../../src/compiler/schedule/schedule_state.js';
export { SRefTree, SRef } from '../../src/compiler/schedule/sref.js';
export { buildBlockScopes, scopeRootSRef } from '../../src/compiler/schedule/block_scope.js';
export { SchedulePolicy, classifyBlock } from '../../src/compiler/schedule/rules.js';
export { ScheduleValidator } from '../../src/compiler/schedule/validator.js';
export { ScheduleTrace } from '../../src/compiler/schedule/trace.js';
export {
  reorderLegality, loopCarriedDependence, IterVarPolicy, reductionLoopVars,
} from '../../src/compiler/schedule/legality.js';

export { lowerGraphToPrimFunc } from '../../src/compiler/passes/lowering/graph_to_tensor.js';
export { printTensorIR } from '../../src/compiler/ir/tensor/printer.js';
export { trace } from '../../src/tracing/compile.js';
export { BackendPipeline } from '../../src/backend/pipeline.js';

export {
  ForNode, BlockNode, BlockRealizeNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, PrimFunc, ForKind, IterVarKind,
} from '../../src/compiler/ir/tensor/nodes.js';
export { Buffer } from '../../src/compiler/ir/tensor/buffer.js';

export {
  dependences, accessDependence, permutationPreservesDependences, Direction, DepKind,
} from '../../src/compiler/analysis/dependence.js';
export { collectBufferAccesses } from '../../src/compiler/analysis/buffer_access.js';
export {
  profileGpuAccesses, launchGeometry, crossBlockRAWBuffers, threadSharedIntermediates,
} from '../../src/compiler/analysis/gpu_race.js';

// --- Part VIII: the autotuner. Nothing in `src/compiler/autotune/` is public;
// the labs drive the search, the cost models and the tuning database directly.

export { enumerateFactorizations } from '../../src/compiler/autotune/factorization.js';
export { getTileStructure, levelCounts, CPU_TILING_SSRSRS } from '../../src/compiler/autotune/tile_structure.js';
export { createMultiLevelTilingSketch, createSSRSRSTilingSketch } from '../../src/compiler/autotune/tiling.js';
export { ScheduleSketch, SearchVariable } from '../../src/compiler/autotune/sketch.js';
export { deriveSketches } from '../../src/compiler/autotune/derivation.js';
export { getSketchesForBlock } from '../../src/compiler/autotune/search_space.js';
export { analyzeBlockStructure, collectAllBlockNames, findBlock } from '../../src/compiler/autotune/block_analysis.js';
export { buildBlockDAG, findFusibleConsumer } from '../../src/compiler/autotune/block_dag.js';
export {
  createElementwiseCPUSketch, createReductionCPUSketch, createRfactorSketch, createFusedTilingSketch,
} from '../../src/compiler/autotune/sketch_generators.js';

export { FeatureExtractor, STATEMENT_FEATURE_SCHEMA } from '../../src/compiler/autotune/features.js';
export { AnalyticalCostModel, LearnedCostModel, GuidedCostModel } from '../../src/compiler/autotune/cost_model.js';
export { GradientBoostedTrees } from '../../src/compiler/autotune/gbt.js';

export { RandomSearch, EvolutionarySearch, SearchCandidate, createSearchStrategy } from '../../src/compiler/autotune/search.js';
export { BenchmarkRunner, BenchmarkResult, robustStats } from '../../src/compiler/autotune/benchmark.js';
export { Deadline } from '../../src/compiler/autotune/budget.js';
export { TaskScheduler, GradientSchedulerPolicy } from '../../src/compiler/autotune/task_scheduler.js';
export { TuningDatabase, TuningRecord, CODEGEN_VERSION } from '../../src/compiler/autotune/tuning_db.js';
export { computeWorkloadKey, buildBlockMap } from '../../src/compiler/autotune/workload_key.js';
export { Autotuner } from '../../src/compiler/autotune/autotuner.js';
export { BlockTuningSession, gpuThreadBlockSize } from '../../src/compiler/autotune/session.js';
export { clonePrimFunc, extractBlockMini } from '../../src/compiler/autotune/tune_ir.js';
export { matmulTileDims, analyzePureMatmul, pickFixedConfig } from '../../src/compiler/autotune/gpu_matmul_sketch.js';

export { ScheduleStep } from '../../src/compiler/schedule/trace.js';
export { compileGraph } from '../../src/compiler/pipeline/compiler.js';
export { buildFunction } from '../../src/compiler/ir/graph/builder.js';
export { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
