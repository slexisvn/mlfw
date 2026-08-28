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

export { lowerToLIR } from '../../src/compiler/passes/lowering/tensor_to_lir.js';
export { detectAccumulator, ACCUMULATOR_OPS } from '../../src/compiler/passes/lowering/accumulator.js';
export { scanMetadata } from '../../src/compiler/ir/lir/scanner.js';
export { flattenIndex, computeDynamicStride, computeNumelExpr } from '../../src/compiler/ir/lir/flatten.js';
export { verifyLIR, LIRVerificationError } from '../../src/compiler/ir/lir/verifier.js';
export { buildLirPipeline } from '../../src/compiler/pipeline/lir_pipeline.js';
export { LirPassManager } from '../../src/compiler/passes/lir_pass_manager.js';
export { FlatIndexSimplifyPass } from '../../src/compiler/passes/simplify/flat_index_simplify.js';
export {
  LIRFunc, LIRFlatLoadNode, LIRFlatStoreNode, LIRAccumulatorNode, LIRBindingsNode,
  LIRMetadata, inferDtype, normalizeDtype, isWasmNativeOp,
} from '../../src/compiler/ir/lir/nodes.js';

export { CPUCodegen } from '../../src/backend/cpu/codegen.js';
export { WasmCodegen } from '../../src/backend/wasm/codegen.js';
export { CUDACodegen } from '../../src/backend/cuda/codegen.js';
export { WebGPUCodegen } from '../../src/backend/webgpu/codegen.js';
export { encodeWat } from '../../src/backend/wasm/wat_encoder.js';
export { flattenRowMajorIndex } from '../../src/backend/index_emit.js';
export { emitSymInt } from '../../src/backend/codegen_utils.js';
export { TargetKind } from '../../src/backend/target.js';
export {
  registerCodegen, getCodegenEntry,
  registerExternalCodegen, getExternalCodegen, unregisterExternalCodegen,
} from '../../src/backend/codegen_registry.js';
export { getCudaIntrin, registerCudaIntrin } from '../../src/backend/cuda/tensor_intrin.js';

export {
  detectPureMatmul, CUBLAS_PROVIDER, isExternalCodegenEnabled,
  registerExternalCodegenProvider, unregisterExternalCodegenProvider,
  activeExternalCodegenProviders,
} from '../../src/compiler/pipeline/external_codegen.js';
export { FuncAttr } from '../../src/compiler/ir/func_attrs.js';

export { CompiledKernel } from '../../src/backend/pipeline.js';
export { RuntimeModule, RuntimeTensor, KernelRegistry, constBuffersOf } from '../../src/runtime/runtime.js';
export { registerBackend, getBackend, hasBackend } from '../../src/runtime/backend_registry.js';
export {
  assignPlanBuffers, computePlanDonations, planMemoryReport,
} from '../../src/compiler/passes/memory/plan_buffer_assignment.js';

export {
  DispatchKey, DispatchKeySet, EMPTY_KEY_SET, BACKEND_KEY_SET, AUTOGRAD_KEY_SET,
  FUNCTIONALITY_KEY_SET, backendKeyForDevice, autogradKeyForBackend,
} from '../../src/dispatcher/dispatch_key.js';
export { computeKeySet } from '../../src/dispatcher/dispatcher.js';
export { KernelFunction, IValue, IValueTag } from '../../src/dispatcher/boxing.js';
export { KernelTable } from '../../src/dispatcher/kernel_table.js';
export { OperatorEntry } from '../../src/dispatcher/operator_entry.js';
export { OperatorHandle } from '../../src/dispatcher/operator_handle.js';
export { Library } from '../../src/dispatcher/library.js';
export { guardStack, withExcludedKeys, withIncludedKeys, withGuard } from '../../src/dispatcher/guard.js';
export { parseSchema, OperatorSchema, SchemaArg, ArgKind } from '../../src/dispatcher/operator_schema.js';
export { jitCompile, jitCacheClear } from '../../src/dispatcher/jit_cache.js';

export { Tracer, getActiveTracer } from '../../src/tracing/tracer.js';
export { registerTracingDispatch } from '../../src/tracing/dispatch.js';
export { ShapeEnv } from '../../src/tracing/shape_env.js';
export { _traceCore } from '../../src/tracing/compile.js';
export { foldWeightParams, weightPredicate, MAX_FOLDABLE_ELEMENTS } from '../../src/tracing/fold_params.js';
export { SymInt } from '../../src/compiler/analysis/sym_int.js';
export {
  BASELINE, DEFAULT_MIN_GAIN, optimizationCandidates, selectWinner,
  candidateByName, gateCacheKey, graphSignature,
} from '../../src/compiler/pipeline/opt_gate.js';
