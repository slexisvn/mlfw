import { noGrad, onesLike } from 'mlfw/index.js';
import { targetNote } from '../catalog/targets.js';
import { instrumentLayers } from './layer_sites.js';
import { modelLines } from './source_map.js';
import { describe, previewsOf, valuesOf } from './tensor_stats.js';
import type { TensorLike as Numeric } from './tensor_stats.js';
import type { BackwardMode, LayerActivation, RunResult, TargetName } from '../protocol.js';

const WARMUP = 2;
const MIN_BATCH_MS = 25;
const MAX_TOTAL_MS = 400;
const MAX_ITERATIONS = 4096;

type TensorLike = Numeric & {
  grad?: TensorLike | null;
  requiresGrad_?: (flag?: boolean) => TensorLike;
  backward?: (grad?: unknown) => void;
};

type Trainable = {
  (...inputs: unknown[]): unknown;
  backward(...gradOutputs: unknown[]): unknown;
  capturedParams(): TensorLike[];
};

export type Model = {
  forward(...args: never[]): unknown;
  namedParameters?: () => Iterable<[string, unknown]>;
};

const EMPTY: RunResult = {
  ran: false, skipped: null, error: null,
  inputs: [], outputs: [], eagerOutputs: [],
  gradients: [], eagerGradients: [],
  parameters: [], layers: [],
  maxAbsDiff: null, maxAbsGradDiff: null,
  compiledMs: null, eagerMs: null, iterations: 0,
};

function asTensors(output: unknown): TensorLike[] {
  return (Array.isArray(output) ? output : [output]) as TensorLike[];
}

const inputName = (index: number): string => `input ${index}`;
const outputName = (index: number): string => `output ${index}`;

function paramNames(model: Model, params: readonly TensorLike[]): string[] {
  const named = new Map<unknown, string>();
  if (typeof model.namedParameters === 'function') {
    for (const [name, param] of model.namedParameters()) named.set(param, name);
  }
  return params.map((param, i) => named.get(param) ?? `param ${i}`);
}

function diffOf(x: number, y: number): number {
  if (Object.is(x, y)) return 0;
  if (Number.isNaN(x) || Number.isNaN(y)) return Infinity;
  const diff = Math.abs(x - y);
  return Number.isFinite(diff) ? diff : Infinity;
}

function maxAbsDiff(a: readonly number[][], b: readonly number[][]): number | null {
  if (a.length !== b.length) return null;
  let worst = 0;

  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return null;
    for (let j = 0; j < a[i].length; j++) {
      const diff = diffOf(a[i][j], b[i][j]);
      if (diff > worst) worst = diff;
    }
  }

  return worst;
}

async function layerActivations(model: Model, inputs: readonly TensorLike[]): Promise<LayerActivation[]> {
  const { rows, stop } = instrumentLayers(model);
  try {
    await noGrad(() => model.forward(...(inputs as never[])));
  } catch {
    return [];
  } finally {
    stop();
  }

  return rows.map(row => ({
    name: row.name,
    kind: row.kind,
    line: modelLines(row.site)[0] ?? null,
    outputs: row.outputs.map((tensor, i) => {
      const numeric = tensor as Numeric;
      return describe(row.outputs.length > 1 ? `out ${i}` : '', numeric, valuesOf(numeric));
    }),
  }));
}

async function timed(call: () => unknown): Promise<{ output: unknown; ms: number; iterations: number }> {
  for (let i = 0; i < WARMUP; i++) await call();

  const deadline = performance.now() + MAX_TOTAL_MS;
  let iterations = 1;
  let output: unknown;
  let batchMs = 0;

  for (;;) {
    const started = performance.now();
    for (let i = 0; i < iterations; i++) output = await call();
    batchMs = performance.now() - started;

    if (batchMs >= MIN_BATCH_MS || iterations >= MAX_ITERATIONS || performance.now() >= deadline) break;
    iterations *= 2;
  }

  return { output, ms: batchMs / iterations, iterations };
}

async function seededOnes(compiled: (...args: unknown[]) => unknown, inputs: readonly unknown[]): Promise<TensorLike[]> {
  return asTensors(await compiled(...inputs)).map(tensor => onesLike(tensor as never) as TensorLike);
}

function eagerStep(model: Model, inputs: readonly TensorLike[], params: readonly TensorLike[], seeds: readonly TensorLike[]) {
  return async (): Promise<{ outputs: TensorLike[]; grads: TensorLike[] }> => {
    for (const tensor of [...inputs, ...params]) {
      tensor.requiresGrad_?.(true);
      tensor.grad = null;
    }
    const outputs = asTensors(await model.forward(...(inputs as never[])));
    for (let i = 0; i < outputs.length; i++) outputs[i].backward?.(seeds[i]);
    return { outputs, grads: [...inputs, ...params].map(tensor => tensor.grad as TensorLike) };
  };
}

async function runInference(
  compiled: (...inputs: unknown[]) => unknown,
  model: Model,
  inputs: readonly TensorLike[],
): Promise<RunResult> {
  const run = await timed(() => compiled(...inputs));
  const eager = await timed(() => noGrad(() => model.forward(...(inputs as never[]))));

  const compiledOut = previewsOf(asTensors(run.output), outputName);
  const eagerOut = previewsOf(asTensors(eager.output), outputName);

  return {
    ...EMPTY,
    ran: true,
    inputs: previewsOf(inputs, inputName).previews,
    outputs: compiledOut.previews,
    eagerOutputs: eagerOut.previews,
    layers: await layerActivations(model, inputs),
    maxAbsDiff: maxAbsDiff(compiledOut.values, eagerOut.values),
    compiledMs: run.ms,
    eagerMs: eager.ms,
    iterations: run.iterations,
  };
}

async function runTraining(
  handle: Trainable,
  model: Model,
  inputs: readonly TensorLike[],
): Promise<RunResult> {
  const seeds = await seededOnes(handle, inputs);
  const params = handle.capturedParams();

  const run = await timed(async () => {
    const outputs = asTensors(await handle(...inputs));
    const grads = asTensors(await handle.backward(...seeds));
    return { outputs, grads };
  });

  const eager = await timed(eagerStep(model, inputs, params, seeds));

  const compiledStep = run.output as { outputs: TensorLike[]; grads: TensorLike[] };
  const eagerStepOut = eager.output as { outputs: TensorLike[]; grads: TensorLike[] };

  const names = paramNames(model, params);
  const gradName = (index: number): string => (
    index < inputs.length ? `d ${inputName(index)}` : `d ${names[index - inputs.length]}`
  );

  const compiledOut = previewsOf(compiledStep.outputs, outputName);
  const eagerOut = previewsOf(eagerStepOut.outputs, outputName);
  const compiledGrads = previewsOf(compiledStep.grads, gradName);
  const eagerGrads = previewsOf(eagerStepOut.grads, gradName);

  return {
    ...EMPTY,
    ran: true,
    inputs: previewsOf(inputs, inputName).previews,
    outputs: compiledOut.previews,
    eagerOutputs: eagerOut.previews,
    gradients: compiledGrads.previews,
    eagerGradients: eagerGrads.previews,
    parameters: previewsOf(params, i => names[i]).previews,
    layers: await layerActivations(model, inputs),
    maxAbsDiff: maxAbsDiff(compiledOut.values, eagerOut.values),
    maxAbsGradDiff: maxAbsDiff(compiledGrads.values, eagerGrads.values),
    compiledMs: run.ms,
    eagerMs: eager.ms,
    iterations: run.iterations,
  };
}

export async function executeCompiled(
  compiled: (...inputs: unknown[]) => unknown,
  model: Model,
  inputs: readonly unknown[],
  target: TargetName,
  backward: BackwardMode,
): Promise<RunResult> {
  const skipped = targetNote(target).skipReason;
  if (skipped) return { ...EMPTY, skipped };

  try {
    return backward === 'off'
      ? await runInference(compiled, model, inputs as TensorLike[])
      : await runTraining(compiled as Trainable, model, inputs as TensorLike[]);
  } catch (error) {
    return { ...EMPTY, error: error instanceof Error ? error.message : String(error) };
  }
}
