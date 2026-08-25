import { noGrad } from 'mlfw/index.js';
import { targetNote } from '../catalog/targets.js';
import type { RunResult, TargetName, TensorPreview } from '../protocol.js';

const WARMUP = 2;
const MIN_BATCH_MS = 25;
const MAX_TOTAL_MS = 400;
const MAX_ITERATIONS = 4096;
const PREVIEW_VALUES = 8;

type TensorLike = {
  shape?: readonly number[];
  dtype?: unknown;
  data?: ArrayLike<number>;
  contiguous?: () => { data: ArrayLike<number> };
};

function asTensors(output: unknown): TensorLike[] {
  return (Array.isArray(output) ? output : [output]) as TensorLike[];
}

function valuesOf(tensor: TensorLike): number[] {
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

export async function executeCompiled(
  compiled: (...args: unknown[]) => unknown,
  model: { forward(...args: never[]): unknown },
  inputs: readonly unknown[],
  target: TargetName,
): Promise<RunResult> {
  const empty: RunResult = {
    ran: false, skipped: null, error: null,
    inputs: [], outputs: [], eagerOutputs: [],
    maxAbsDiff: null, compiledMs: null, eagerMs: null, iterations: 0,
  };

  const skipped = targetNote(target).skipReason;
  if (skipped) return { ...empty, skipped };

  try {
    const run = await timed(() => compiled(...inputs));
    const eager = await timed(() => noGrad(() => model.forward(...(inputs as never[]))));

    const compiledTensors = asTensors(run.output);
    const eagerTensors = asTensors(eager.output);
    const compiledValues = compiledTensors.map(valuesOf);
    const eagerValues = eagerTensors.map(valuesOf);

    return {
      ran: true,
      skipped: null,
      error: null,
      inputs: (inputs as TensorLike[]).map(tensor => describe(tensor, valuesOf(tensor))),
      outputs: compiledTensors.map((tensor, i) => describe(tensor, compiledValues[i])),
      eagerOutputs: eagerTensors.map((tensor, i) => describe(tensor, eagerValues[i])),
      maxAbsDiff: maxAbsDiff(compiledValues, eagerValues),
      compiledMs: run.ms,
      eagerMs: eager.ms,
      iterations: run.iterations,
    };
  } catch (error) {
    return { ...empty, error: error instanceof Error ? error.message : String(error) };
  }
}
