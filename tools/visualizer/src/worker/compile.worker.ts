import { compile, manual_seed, CPUTarget, WasmTarget, CUDATarget, WebGPUTarget, TraceLevel } from 'mlfw/index.js';
import { PassContext } from 'mlfw/compiler/passes/pass.js';
import { CompileRecorder } from './recorder.js';
import { executeCompiled } from './execute.js';
import { evaluateModelSource, frameworkGlobals } from './evaluate.js';
import { recordSourceLines } from './source_map.js';
import type { CompileOptions, CompileResponse, Kernel, RunResult, SourceLink, TargetName, WorkerRequest } from '../protocol.js';

const NOT_RUN: RunResult = {
  ran: false, skipped: null, error: null,
  inputs: [], outputs: [], eagerOutputs: [],
  maxAbsDiff: null, compiledMs: null, eagerMs: null, iterations: 0,
};

const TARGETS: Record<TargetName, () => unknown> = {
  cpu: CPUTarget,
  wasm: WasmTarget,
  cuda: CUDATarget,
  webgpu: WebGPUTarget,
};

const KERNEL_LANGUAGE: Record<TargetName, string> = {
  cpu: 'javascript',
  wasm: 'wat',
  cuda: 'cpp',
  webgpu: 'wgsl',
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
    target: TARGETS[options.target](),
    fusion: { enabled: options.fusion, strategy: options.fusionStrategy },
    scheduling: { enabled: options.scheduling },
    optimization: { layout: options.layout },
    passContext: options.disabledPasses.length > 0
      ? new PassContext({ disabledPasses: options.disabledPasses })
      : null,
  };
}

function collectKernels(handle: { result(): unknown }, target: TargetName): Kernel[] {
  const result = handle.result() as {
    listKernels(): string[];
    getSource(name: string): string | null;
  } | null;
  if (!result) return [];

  const kernels: Kernel[] = [];
  for (const name of result.listKernels()) {
    const source = result.getSource(name);
    if (source === null) continue;
    kernels.push({ name, source, language: KERNEL_LANGUAGE[target] });
  }
  return kernels;
}

async function runCompile(id: number, source: string, options: CompileOptions): Promise<CompileResponse> {
  const startedAt = performance.now();
  let error: string | null = null;
  let errorPhase: string | null = null;
  let kernels: Kernel[] = [];
  let run: RunResult = NOT_RUN;
  let sourceLinks: SourceLink[] = [];
  let stopRecordingLines = (): void => {};
  const recorder = new CompileRecorder(() => { stopRecordingLines(); });

  try {
    manual_seed(SEED);
    const { model, inputs, baseLine } = evaluateModelSource(source);
    const lineRecorder = recordSourceLines(baseLine);
    stopRecordingLines = lineRecorder.stop;
    const compilable = asCompilable(model);
    const handle = compile(compilable, inputs as never[], {
      ...compilerOptions(options),
      instruments: [recorder],
      trace: { level: TraceLevel.DEBUG, sink: recorder.sink },
    } as never) as ((...args: unknown[]) => unknown) & { _ready: Promise<void> | null; result(): unknown };

    if (handle._ready) await handle._ready;
    kernels = collectKernels(handle, options.target);
    manual_seed(SEED);
    run = await executeCompiled(handle, compilable, inputs, options.target);
    sourceLinks = [...lineRecorder.lines];
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
    errorPhase = recorder.currentOpenPass();
    recorder.closeOpenSteps();
  } finally {
    stopRecordingLines();
  }

  return {
    kind: 'compile',
    id,
    ok: error === null,
    error,
    errorPhase,
    steps: recorder.timeline(kernels),
    kernels,
    events: recorder.events,
    sourceLinks,
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
