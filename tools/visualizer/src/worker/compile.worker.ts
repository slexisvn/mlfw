import { compile, compileWithBackward, manual_seed, TraceLevel } from 'mlfw/index.js';
import { CompileRecorder } from './recorder.js';
import { asCompilable, compilerOptions } from './settings.js';
import { executeCompiled } from './execute.js';
import { evaluateModelSource, frameworkGlobals } from './evaluate.js';
import { attributeLayerSites } from './layer_sites.js';
import { collectDagLines } from './snapshot.js';
import { bisect } from './bisect.js';
import { semanticReport } from './semantics.js';
import { kernelReports } from './kernel_report.js';
import { targetNote } from '../catalog/targets.js';
import type { BackwardMode, CompileOptions, CompileResponse, CompileStep, Kernel, LaunchDiagnosis, RunResult, SemanticsResponse, TargetName, WorkerRequest } from '../protocol.js';

const NOT_RUN: RunResult = {
  ran: false, skipped: null, error: null,
  inputs: [], outputs: [], eagerOutputs: [],
  gradients: [], eagerGradients: [],
  parameters: [], layers: [],
  maxAbsDiff: null, maxAbsGradDiff: null,
  compiledMs: null, eagerMs: null, iterations: 0,
};

const SEED = 0;

let lastRecorder: CompileRecorder | null = null;

type KernelSource = {
  listKernels(): string[];
  getSource(name: string): string | null;
  getMetadata?(name: string): Record<string, unknown> | null;
  module?: { getKernelMetadata(name: string): Record<string, unknown> | null };
};
type InferenceHandle = ((...args: unknown[]) => unknown) & { _ready: Promise<void> | null; result(): unknown };
type TrainingHandle = ((...args: unknown[]) => unknown) & { compiledUnits(): { name: string; result: unknown }[] };

function metadataOf(result: KernelSource, name: string): Record<string, unknown> | null {
  if (typeof result.getMetadata === 'function') return result.getMetadata(name);
  if (result.module) return result.module.getKernelMetadata(name);
  return null;
}

function kernelsOf(result: KernelSource | null, language: string, prefix: string): Kernel[] {
  if (!result) return [];
  const kernels: Kernel[] = [];
  for (const name of result.listKernels()) {
    const source = result.getSource(name);
    if (source === null) continue;
    const metadata = metadataOf(result, name);
    kernels.push({
      name: prefix ? `${prefix} · ${name}` : name,
      source,
      language,
      metadata,
      diagnosis: (metadata?.launchDiagnosis ?? null) as LaunchDiagnosis | null,
    });
  }
  return kernels;
}

function collectKernels(handle: InferenceHandle | TrainingHandle, target: TargetName, backward: BackwardMode): Kernel[] {
  const language = targetNote(target).kernelLanguage;
  if (backward === 'off') {
    return kernelsOf((handle as InferenceHandle).result() as KernelSource | null, language, '');
  }
  const units = (handle as TrainingHandle).compiledUnits();
  return units.flatMap(unit => kernelsOf(unit.result as KernelSource, language, unit.name));
}

function tracedLines(steps: readonly CompileStep[]): number[] {
  const lines = new Set<number>();
  for (const step of steps) {
    for (const side of [step.before, step.after]) {
      for (const dag of side.dags) collectDagLines(dag.nodes, lines);
    }
  }
  return [...lines].sort((a, b) => a - b);
}

async function runCompile(id: number, source: string, options: CompileOptions): Promise<CompileResponse> {
  const startedAt = performance.now();
  let error: string | null = null;
  let errorPhase: string | null = null;
  let kernels: Kernel[] = [];
  let run: RunResult = NOT_RUN;
  let stopRecordingLines = (): void => {};
  let stopLayerSites = (): void => {};
  const recorder = new CompileRecorder(options.verify !== 'off', () => { stopLayerSites(); stopRecordingLines(); });

  try {
    manual_seed(SEED);
    const { model, inputs } = evaluateModelSource(source, stop => { stopRecordingLines = stop; });
    const compilable = asCompilable(model);
    stopLayerSites = attributeLayerSites(model);
    const settings = {
      ...compilerOptions(options),
      instruments: [recorder],
      trace: { level: TraceLevel.DEBUG, sink: recorder.sink },
    };

    const handle = options.backward === 'off'
      ? compile(compilable, inputs as never[], settings as never) as InferenceHandle
      : compileWithBackward(compilable, inputs as never[], { ...settings, mode: options.backward } as never) as TrainingHandle;

    const ready = (handle as InferenceHandle)._ready;
    if (ready) await ready;
    kernels = collectKernels(handle, options.target, options.backward);
    manual_seed(SEED);
    run = await executeCompiled(handle, compilable, inputs, options.target, options.backward);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
    errorPhase = recorder.currentOpenPass();
    recorder.closeOpenSteps();
  } finally {
    stopLayerSites();
    stopRecordingLines();
  }

  const steps = recorder.timeline(kernels);
  lastRecorder = recorder;

  return {
    kind: 'compile',
    id,
    ok: error === null,
    error,
    errorPhase,
    steps,
    kernels,
    events: recorder.events,
    sourceLines: tracedLines(steps),
    memoryPlans: recorder.memoryPlans(),
    tuningRounds: recorder.tuningRounds(),
    skipped: recorder.skippedPasses(),
    kernelReports: kernelReports(kernels),
    totalMs: performance.now() - startedAt,
    run,
  };
}

function semantics(id: number, step: number): SemanticsResponse {
  const startedAt = performance.now();
  const captured = lastRecorder ? lastRecorder.bodies.get(step) : undefined;

  if (!captured) {
    return {
      kind: 'semantics',
      id,
      step,
      report: null,
      unavailable: 'Only a pass over the tensor or low-level IR can be interpreted — a graph pass has no loop nest to run, and a very large function is not kept.',
      ms: performance.now() - startedAt,
    };
  }

  return {
    kind: 'semantics',
    id,
    step,
    report: semanticReport(captured.before, captured.after),
    unavailable: null,
    ms: performance.now() - startedAt,
  };
}

self.onmessage = async (message: MessageEvent<WorkerRequest>) => {
  const request = message.data;

  if (request.kind === 'init') {
    self.postMessage({ kind: 'init', id: request.id, globals: frameworkGlobals() });
    return;
  }

  if (request.kind === 'semantics') {
    self.postMessage(semantics(request.id, request.step));
    return;
  }

  if (request.kind === 'bisect') {
    self.postMessage(await bisect(request, (probe, note) => {
      self.postMessage({ kind: 'bisect-progress', id: request.id, probe, note });
    }));
    return;
  }

  self.postMessage(await runCompile(request.id, request.source, request.options));
};
