import { compile, compileWithBackward, manual_seed, TraceLevel } from 'mlfw/index.js';
import { PassContext } from 'mlfw/compiler/passes/pass.js';
import { CompileRecorder } from './recorder.js';
import { executeCompiled } from './execute.js';
import { evaluateModelSource, frameworkGlobals } from './evaluate.js';
import { attributeLayerSites } from './layer_sites.js';
import { collectDagLines } from './snapshot.js';
import { targetNote } from '../catalog/targets.js';
import { TARGET_FACTORIES } from './targets.js';
import { SEARCH_BUDGET } from '../catalog/tuning.js';
import type { BackwardMode, CompileOptions, CompileResponse, CompileStep, Kernel, RunResult, TargetName, WorkerRequest } from '../protocol.js';

const NOT_RUN: RunResult = {
  ran: false, skipped: null, error: null,
  inputs: [], outputs: [], eagerOutputs: [],
  gradients: [], eagerGradients: [],
  parameters: [], layers: [],
  maxAbsDiff: null, maxAbsGradDiff: null,
  compiledMs: null, eagerMs: null, iterations: 0,
};

const SEED = 0;
const NON_IDENTIFIER = /[^A-Za-z0-9_$]/g;
const FALLBACK_NAME = 'model';

type Compilable = Parameters<typeof compile>[0];
type ForwardFn = (...args: unknown[]) => unknown;

function identifier(name: string): string {
  const cleaned = name.replace(NON_IDENTIFIER, '');
  return /^[A-Za-z_$]/.test(cleaned) ? cleaned : FALLBACK_NAME;
}

function wrap(forward: ForwardFn): Compilable {
  const wrapper = class { forward = forward; };
  Object.defineProperty(wrapper, 'name', { value: identifier(forward.name) });
  return new wrapper() as unknown as Compilable;
}

function asCompilable(model: unknown): Compilable {
  if (model && typeof (model as { forward?: unknown }).forward === 'function') return model as Compilable;
  if (typeof model === 'function') return wrap(model as ForwardFn);
  throw new Error('run(model, inputs): model must be an nn.Module or a function');
}

function compilerOptions(options: CompileOptions): Record<string, unknown> {
  return {
    target: TARGET_FACTORIES[options.target](),
    fusion: { enabled: options.fusion, strategy: options.fusionStrategy },
    scheduling: { enabled: options.scheduling, autotune: options.autotune, ...SEARCH_BUDGET },
    optimization: { layout: options.layout },
    passContext: options.disabledPasses.length > 0
      ? new PassContext({ disabledPasses: options.disabledPasses })
      : null,
  };
}

type KernelSource = { listKernels(): string[]; getSource(name: string): string | null };
type InferenceHandle = ((...args: unknown[]) => unknown) & { _ready: Promise<void> | null; result(): unknown };
type TrainingHandle = ((...args: unknown[]) => unknown) & { compiledUnits(): { name: string; result: unknown }[] };

function kernelsOf(result: KernelSource | null, language: string, prefix: string): Kernel[] {
  if (!result) return [];
  const kernels: Kernel[] = [];
  for (const name of result.listKernels()) {
    const source = result.getSource(name);
    if (source === null) continue;
    kernels.push({ name: prefix ? `${prefix} · ${name}` : name, source, language });
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
  const recorder = new CompileRecorder(() => { stopLayerSites(); stopRecordingLines(); });

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
    totalMs: performance.now() - startedAt,
    run,
  };
}

self.onmessage = async (message: MessageEvent<WorkerRequest>) => {
  const request = message.data;

  if (request.kind === 'init') {
    self.postMessage({ kind: 'init', id: request.id, globals: frameworkGlobals() });
    return;
  }

  self.postMessage(await runCompile(request.id, request.source, request.options));
};
