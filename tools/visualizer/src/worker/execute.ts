import { noGrad, onesLike } from 'mlfw/index.js';
import { targetNote } from '../catalog/targets.js';
import type { BackwardMode, RunResult, TargetName, TensorPreview } from '../protocol.js';

const WARMUP = 2;
const MIN_BATCH_MS = 25;
const MAX_TOTAL_MS = 400;
const MAX_ITERATIONS = 4096;
const PREVIEW_VALUES = 8;

type TensorLike = {
  shape?: readonly number[];
  dtype?: unknown;
  data?: ArrayLike<number>;
  grad?: TensorLike | null;
  contiguous?: () => { data: ArrayLike<number> };
  requiresGrad_?: (flag?: boolean) => TensorLike;
  backward?: (grad?: unknown) => void;
};

type Trainable = {
  (...inputs: unknown[]): unknown;
  backward(...gradOutputs: unknown[]): unknown;
  capturedParams(): TensorLike[];
};

export type Model = { forward(...args: never[]): unknown };

const EMPTY: RunResult = {
  ran: false, skipped: null, error: null,
  inputs: [], outputs: [], eagerOutputs: [],
  gradients: [], eagerGradients: [],
  maxAbsDiff: null, maxAbsGradDiff: null,
  compiledMs: null, eagerMs: null, iterations: 0,
};

function asTensors(output: unknown): TensorLike[] {
  return (Array.isArray(output) ? output : [output]) as TensorLike[];
}

function valuesOf(tensor: TensorLike | null | undefined): number[] {
  if (!tensor) return [];
  const array = tensor.contiguous ? tensor.contiguous().data : tensor.data;
  if (!array) return [];
  const values = new Array<number>(array.length);
  for (let i = 0; i < array.length; i++) values[i] = Number(array[i]);
  return values;
}

function describe(tensor: TensorLike, values: readonly number[]): TensorPreview {
  return {
    shape: [...(tensor.shape ?? [])],
    dtype: String(tensor.dtype ?? 'f32'),
    numel: values.length,
    preview: values.slice(0, PREVIEW_VALUES),
  };
}

function previews(tensors: readonly TensorLike[]): { previews: TensorPreview[]; values: number[][] } {
  const values = tensors.map(valuesOf);
  return { previews: tensors.map((tensor, i) => describe(tensor, values[i])), values };
}

function maxAbsDiff(a: readonly number[][], b: readonly number[][]): number | null {
  if (a.length !== b.length) return null;
  let worst = 0;

  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return null;
    for (let j = 0; j < a[i].length; j++) {
      const diff = Math.abs(a[i][j] - b[i][j]);
      if (Number.isFinite(diff) && diff > worst) worst = diff;
    }
  }

  return worst;
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

  const compiledOut = previews(asTensors(run.output));
  const eagerOut = previews(asTensors(eager.output));

  return {
    ...EMPTY,
    ran: true,
    inputs: previews(inputs).previews,
    outputs: compiledOut.previews,
    eagerOutputs: eagerOut.previews,
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

  const compiledOut = previews(compiledStep.outputs);
  const eagerOut = previews(eagerStepOut.outputs);
  const compiledGrads = previews(compiledStep.grads);
  const eagerGrads = previews(eagerStepOut.grads);

  return {
    ...EMPTY,
    ran: true,
    inputs: previews(inputs).previews,
    outputs: compiledOut.previews,
    eagerOutputs: eagerOut.previews,
    gradients: compiledGrads.previews,
    eagerGradients: eagerGrads.previews,
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
