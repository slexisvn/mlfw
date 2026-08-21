// Part VII's labs drive the scheduling primitives by hand, and `Schedule` is
// not exported from the package. This module bundles the internal surface
// listed in `docs/tools/internals-entry.ts` with esbuild — a devDependency the
// repository already has — and re-exports it, so a lab is still one command:
//
//     node docs/part7/ch40-loop-primitives/labs/01-split-fuse-reorder.mjs
//
// The bundle lands in the OS temp directory and takes about a tenth of a second
// to build. Nothing is written inside the repository.

import { build } from 'esbuild';
import { renameSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outfile = join(tmpdir(), 'mlfw-book-part7-internals.mjs');
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

// The scheduling language.
export const {
  Schedule, resetVarCounter, ScheduleState, ScheduleTrace, ScheduleValidator,
  SchedulePolicy, classifyBlock, SRefTree, SRef, buildBlockScopes, scopeRootSRef,
  reorderLegality, loopCarriedDependence, IterVarPolicy, reductionLoopVars,
} = m;

// Everything needed to obtain a `PrimFunc`, print it, and run it.
export const { lowerGraphToPrimFunc, printTensorIR, BackendPipeline } = m;

// TIR node constructors, for the hand-built counterexamples.
export const {
  ForNode, BlockNode, BlockRealizeNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, PrimFunc, ForKind, IterVarKind, Buffer,
} = m;

// The analyses the legality rules consult.
export const {
  dependences, accessDependence, permutationPreservesDependences, Direction, DepKind,
  collectBufferAccesses, profileGpuAccesses, launchGeometry,
  crossBlockRAWBuffers, threadSharedIntermediates,
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
