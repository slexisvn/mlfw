// Part VIII's labs drive the autotuner by hand, and nothing under
// `src/compiler/autotune/` is part of the package's public surface — a user
// asks for `scheduling: { autotune: true }` and never sees a sketch, a cost
// model or a workload key. This module bundles the internal surface listed in
// `docs/tools/internals-entry.ts` with esbuild — a devDependency the repository
// already has — and re-exports it, so a lab is still one command:
//
//     node docs/part8/ch44-how-big-is-the-search-space/labs/01-counting-the-space.mjs
//
// The bundle lands in the OS temp directory and takes about a tenth of a second
// to build. Nothing is written inside the repository.

import { build } from 'esbuild';
import { renameSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outfile = join(tmpdir(), 'mlfw-book-part8-internals.mjs');
const staging = `${outfile}.${process.pid}.tmp`;

await build({
  entryPoints: [join(here, '..', 'tools', 'internals-entry.ts')],
  outfile: staging,
  bundle: true,
  format: 'esm',
  platform: 'node',
  external: ['koffi', 'webgpu'],
  logLevel: 'error',
});
renameSync(staging, outfile);

const m = await import(pathToFileURL(outfile).href);

// The public surface the other parts' labs use.
export const {
  compile, trace, CPUTarget, CUDATarget, WasmTarget, WebGPUTarget,
  TraceLevel, randn, zeros, ones, tensor, manual_seed,
} = m;

// The scheduling language of Part VII, which is the alphabet Part VIII searches.
export const {
  Schedule, resetVarCounter, ScheduleState, ScheduleTrace, ScheduleStep,
  ScheduleValidator, SchedulePolicy, classifyBlock,
} = m;

// Everything needed to obtain a `PrimFunc`, print it, and run it.
export const { lowerGraphToPrimFunc, printTensorIR, BackendPipeline } = m;

// The graph-level entry the autotuner actually runs behind.
export const { compileGraph, buildFunction, TensorType, ScalarType } = m;

// TIR node constructors, for the hand-built fixtures.
export const {
  ForNode, BlockNode, BlockRealizeNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, PrimFunc, ForKind, IterVarKind, Buffer,
} = m;

// The search space: factorizations, tile structures, sketches and their derivation.
export const {
  enumerateFactorizations, getTileStructure, levelCounts, CPU_TILING_SSRSRS,
  createMultiLevelTilingSketch, createSSRSRSTilingSketch,
  createElementwiseCPUSketch, createReductionCPUSketch, createRfactorSketch, createFusedTilingSketch,
  ScheduleSketch, SearchVariable, deriveSketches, getSketchesForBlock,
  analyzeBlockStructure, collectAllBlockNames, findBlock,
  buildBlockDAG, findFusibleConsumer,
  matmulTileDims, analyzePureMatmul, pickFixedConfig,
} = m;

// The cost models and their two disjoint feature sets.
export const {
  FeatureExtractor, STATEMENT_FEATURE_SCHEMA,
  AnalyticalCostModel, LearnedCostModel, GuidedCostModel, GradientBoostedTrees,
} = m;

// Search, measurement, budget and the tuning database.
export const {
  RandomSearch, EvolutionarySearch, SearchCandidate, createSearchStrategy,
  BenchmarkRunner, BenchmarkResult, robustStats, Deadline,
  TaskScheduler, GradientSchedulerPolicy,
  TuningDatabase, TuningRecord, CODEGEN_VERSION,
  computeWorkloadKey, buildBlockMap,
  Autotuner, BlockTuningSession, gpuThreadBlockSize,
  clonePrimFunc, extractBlockMini,
} = m;

/** Trace `fn`, lower the traced graph, and hand back a fresh `PrimFunc`. */
export async function lowerToTir(fn, inputs, target = CPUTarget()) {
  resetVarCounter();
  const graph = await trace(fn, inputs);
  const [name] = graph._functions.keys();
  return lowerGraphToPrimFunc(graph._functions.get(name), target);
}

/** Compile a `PrimFunc` all the way to a callable CPU kernel. */
export function toKernel(primFunc, target = CPUTarget()) {
  const source = new BackendPipeline(target).compile(primFunc).source;
  return { source, call: new Function(`return ${source}`)() };
}
