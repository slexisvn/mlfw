import { Tracer } from './tracer.js';
import { registerTracingDispatch } from './dispatch.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { withIncludedKeys } from '../dispatcher/guard.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { SymbolicTensor } from './symbolic_tensor.js';
import type { CompileOptions, DynamicShapes, InputSignature, MaybePromise, TensorOutput, TracedCore } from './types.js';

let _tracingRegistered = false;

export type TraceFunction = (...inputs: SymbolicTensor[]) => MaybePromise<TensorOutput | TensorOutput[]>;

export function isThenable<T>(value: MaybePromise<T>): value is Promise<T> {
  return typeof (value as { then?: unknown }).then === 'function';
}

function _ensureTracing(): void {
  if (!_tracingRegistered) {
    registerTracingDispatch();
    _tracingRegistered = true;
  }
}

export function resolveDynamicShapes(opts?: CompileOptions): DynamicShapes {
  return opts?.dynamicShapes ?? opts?.dynamic_shapes ?? null;
}

function _normalizeDynamicShapes(dynamicShapes: DynamicShapes, exampleInputs: readonly Tensor[]): (Set<number> | null)[] {
  if (!dynamicShapes) return new Array(exampleInputs.length).fill(null);

  const result = new Array<Set<number> | null>(exampleInputs.length);
  for (let i = 0; i < exampleInputs.length; i++) {
    const spec = dynamicShapes[i];
    if (spec === true) {
      const all = new Set<number>();
      for (let d = 0; d < exampleInputs[i].shape.length; d++) all.add(d);
      result[i] = all;
    } else if (spec instanceof Set) {
      result[i] = spec;
    } else {
      result[i] = null;
    }
  }
  return result;
}

export function _traceCore(fn: TraceFunction, exampleInputs: readonly Tensor[], opts?: CompileOptions): MaybePromise<TracedCore> {
  _ensureTracing();

  const name = opts?.name || fn.name || 'traced';
  const tracer = new Tracer(name);
  const dynamicShapes = _normalizeDynamicShapes(resolveDynamicShapes(opts), exampleInputs);

  for (let i = 0; i < exampleInputs.length; i++) {
    tracer.createInput(exampleInputs[i].shape, exampleInputs[i].dtype, dynamicShapes[i]);
  }

  const numUserInputs = exampleInputs.length;
  const symbolicInputs = tracer._initGraph();

  function _finalize(result: TensorOutput | TensorOutput[]): TracedCore {
    if (Array.isArray(result)) {
      tracer.markOutputs(result as SymbolicTensor[]);
    } else {
      tracer.markOutput(result);
    }
    tracer.deactivate();

    const graph = tracer.getGraphModule();
    const func = graph.functions().next().value;

    return {
      graph,
      capturedParams: [...tracer.capturedParams],
      numUserInputs,
      outputTypes: func.outputTypes,
      shapeEnv: tracer.shapeEnv,
      outputSymShapes: tracer.outputSymShapes,
    };
  }

  tracer.activate();
  const TRACING_KEYS = DispatchKeySet.fromKey(DispatchKey.TRACING);
  const result = withIncludedKeys(TRACING_KEYS, () => fn(...symbolicInputs)) as MaybePromise<TensorOutput | TensorOutput[]>;

  if (isThenable(result)) {
    return result.then(
      (resolved: TensorOutput | TensorOutput[]) => _finalize(resolved),
      (error: unknown) => { tracer.deactivate(); throw error; },
    );
  }

  try {
    return _finalize(result);
  } catch (error) {
    tracer.deactivate();
    throw error;
  }
}

export function inputSignatureOf(inputs: readonly Tensor[]): InputSignature {
  return inputs.map(t => ({ dtype: String(t.dtype), device: String(t.device ?? 'cpu') }));
}

export function signatureMatches(a: InputSignature, b: InputSignature): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].dtype !== b[i].dtype || a[i].device !== b[i].device) return false;
  }
  return true;
}

