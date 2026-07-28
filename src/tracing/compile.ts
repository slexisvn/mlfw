import { Tracer } from './tracer.js';
import { registerTracingDispatch } from './dispatch.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { withIncludedKeys } from '../dispatcher/guard.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../backend/target.js';
import { tensorToContiguous, wrapResult } from '../dispatcher/jit_dispatch.js';
import { RuntimeTensor } from '../runtime/runtime.js';
import { typedArrayCtor } from '../tensor/types/dtype.js';
import { computeNumel } from '../tensor/utils/shape_utils.js';

import { foldWeightParams } from './fold_params.js';
import { compileWithBackward } from './compile_backward.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { Device } from '../tensor/types/device.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { TensorType } from '../compiler/ir/graph/types.js';
import type { ShapeEnv } from './shape_env.js';
import type { SymbolicTensor } from './symbolic_tensor.js';
import type { CompilableModel, CompiledEntry, CompiledForward, CompiledResult, CompileOptions, DynamicShapes, GraphModuleLike, MaybePromise, RuntimeArg, RuntimeTensorLike, SymbolicShape, TensorOutput, TracedCore } from './types.js';

let _tracingRegistered = false;

type TraceFunction = (...inputs: SymbolicTensor[]) => MaybePromise<TensorOutput | TensorOutput[]>;
type ExecutionPrepared = {
  funcName: string;
  device: Device;
  outputTypes: readonly TensorType[];
  outputArrays: NumericTypedArray[];
  outputShapes: readonly number[][];
  allArgs: RuntimeArg[];
};
type ReproError = Error & { repro?: unknown };

function _isThenable<T>(value: MaybePromise<T>): value is Promise<T> {
  return typeof (value as { then?: unknown }).then === 'function';
}

function _ensureTracing(): void {
  if (!_tracingRegistered) {
    registerTracingDispatch();
    _tracingRegistered = true;
  }
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
  const dynamicShapes = _normalizeDynamicShapes(opts?.dynamicShapes, exampleInputs);

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

  if (_isThenable(result)) {
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

export function trace(fn: TraceFunction, exampleInputs: readonly Tensor[], opts?: CompileOptions): MaybePromise<GraphModuleLike> {
  const result = _traceCore(fn, exampleInputs, opts);
  if (_isThenable(result)) {
    return result.then((r: TracedCore) => r.graph);
  }
  return result.graph;
}

function _prepareExecution(compiled: CompiledEntry, inputs: readonly Tensor[], shapeEnv: ShapeEnv): ExecutionPrepared {
  const kernels = compiled.result.listKernels();
  if (kernels.length === 0) throw new Error('No kernels compiled');
  const funcName = kernels[0];

  const device = inputs.length > 0 ? inputs[0].device : 'cpu';

  const inputArrays = new Array<NumericTypedArray>(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    inputArrays[i] = tensorToContiguous(inputs[i]);
  }

  const params = compiled.capturedParams;
  const paramArrays = new Array<NumericTypedArray>(params.length);
  for (let i = 0; i < params.length; i++) {
    paramArrays[i] = tensorToContiguous(params[i]);
  }

  const outputTypes = compiled.outputTypes;
  const outputSymShapes = compiled.outputSymShapes;
  const outputArrays = new Array<NumericTypedArray>(outputTypes.length);
  const outputShapes = new Array<number[]>(outputTypes.length);

  for (let i = 0; i < outputTypes.length; i++) {
    const shape = outputSymShapes && shapeEnv
      ? shapeEnv.resolveSymbolicShape(outputSymShapes[i])
      : outputTypes[i].shape as readonly number[];
    const dtype = outputTypes[i].dtype;
    const numel = computeNumel(shape);
    const Ctor = typedArrayCtor(dtype);
    outputArrays[i] = new Ctor(Math.max(numel, 1));
    outputShapes[i] = [...shape];
  }

  const allArgs = new Array(inputArrays.length + paramArrays.length + outputArrays.length);
  let idx = 0;
  for (let i = 0; i < inputArrays.length; i++) allArgs[idx++] = new RuntimeTensor(inputArrays[i], inputs[i].shape, inputs[i].dtype);
  for (let i = 0; i < paramArrays.length; i++) {
    const rt = new RuntimeTensor(paramArrays[i], params[i].shape, params[i].dtype);
    const impl = params[i]._impl;
    if (impl) (rt as RuntimeTensorLike).resident = { key: impl.storage.rawData, version: impl.version };
    allArgs[idx++] = rt;
  }
  for (let i = 0; i < outputArrays.length; i++) allArgs[idx++] = new RuntimeTensor(outputArrays[i], outputShapes[i], outputTypes[i].dtype);

  return { funcName, device: device as Device, outputTypes, outputArrays, outputShapes, allArgs };
}

function _wrapOutputs(device: Device, outputTypes: readonly TensorType[], outputArrays: readonly NumericTypedArray[], outputShapes: readonly (readonly number[])[]): TensorOutput | TensorOutput[] {
  if (outputTypes.length === 1) {
    return wrapResult(outputArrays[0], outputShapes[0], outputTypes[0].dtype, device);
  }
  const results = new Array<TensorOutput>(outputTypes.length);
  for (let i = 0; i < outputTypes.length; i++) {
    results[i] = wrapResult(outputArrays[i], outputShapes[i], outputTypes[i].dtype, device);
  }
  return results;
}

export function executeCompiled(compiled: CompiledEntry, inputs: readonly Tensor[], shapeEnv: ShapeEnv): MaybePromise<TensorOutput | TensorOutput[]> {
  const { funcName, device, outputTypes, outputArrays, outputShapes, allArgs } =
    _prepareExecution(compiled, inputs, shapeEnv);

  const plan = compiled.result.module.executionPlan;
  if (plan) {
    return compiled.result.module.runPlanAsync(plan, allArgs, { resident: true })
      .then(() => _wrapOutputs(device, outputTypes, outputArrays, outputShapes));
  }

  if (compiled.result.isAsync(funcName)) {
    return compiled.result.runAsync(funcName, ...allArgs)
      .then(() => _wrapOutputs(device, outputTypes, outputArrays, outputShapes));
  }

  compiled.result.run(funcName, ...allArgs);
  return _wrapOutputs(device, outputTypes, outputArrays, outputShapes);
}

export function compile(model: CompilableModel, exampleInputs?: Tensor[], opts: CompileOptions = {}): unknown {
  if (opts?.backward) {
    return compileWithBackward(model, exampleInputs, opts);
  }

  const target = opts?.target ?? CPUTarget();
  const compilerOpts: Record<string, unknown> & { target: unknown; verify: boolean } = { target, verify: false, ...opts };
  const dynamicShapes = opts?.dynamic_shapes || null;
  const shapeBuckets = opts?.shapeBuckets || null;
  const foldWeights = opts?.foldWeights ?? opts?.quantization?.foldWeights ?? false;

  const _cacheEntries: CompiledEntry[] = [];

  function _attachRepro(error: unknown, inputs: readonly Tensor[] | undefined, phase: string): unknown {
    if (!error || typeof error !== 'object' || (error as ReproError).repro) return error;
    try {
      (error as ReproError).repro = {
        name: model.constructor?.name || 'compiled',
        phase,
        target: (target as { name?: string } | null | undefined)?.name,
        inputs: (inputs || []).map(t => ({ shape: t.shape, dtype: t.dtype })),
        config: {
          fusion: compilerOpts.fusion,
          scheduling: compilerOpts.scheduling,
          optimization: compilerOpts.optimization,
          quantization: compilerOpts.quantization,
          dynamicShapes: !!dynamicShapes,
        },
      };
    } catch (_error) { return error; }
    return error;
  }

  function _finalize(traced: TracedCore): CompiledEntry {
    const prepared = foldWeights ? foldWeightParams(traced, tensorToContiguous) : traced;
    const result = new Compiler(compilerOpts).compile(prepared.graph) as CompiledResult;
    return {
      result,
      graph: prepared.graph,
      capturedParams: prepared.capturedParams,
      numUserInputs: prepared.numUserInputs,
      outputTypes: prepared.outputTypes,
      shapeEnv: prepared.shapeEnv,
      outputSymShapes: prepared.outputSymShapes,
    };
  }

  function _compileWith(inputs: Tensor[], dynShapes: DynamicShapes): MaybePromise<CompiledEntry> {
    try {
      const traced = _traceCore(
        (...args: SymbolicTensor[]) => model.forward(...args),
        inputs,
        { name: model.constructor.name || 'compiled', dynamicShapes: dynShapes }
      );
      if (_isThenable(traced)) {
        return traced.then(_finalize, (e: unknown) => { throw _attachRepro(e, inputs, 'compile'); });
      }
      return _finalize(traced);
    } catch (e) {
      throw _attachRepro(e, inputs, 'compile');
    }
  }

  function _compile(inputs: Tensor[]): MaybePromise<CompiledEntry> {
    return _compileWith(inputs, dynamicShapes);
  }

  function _bucketInputs(shapes: readonly (readonly number[])[]): Tensor[] {
    return shapes.map((shape, i) => ({ shape, dtype: exampleInputs![i].dtype } as Tensor));
  }

  function _findCachedEntry(inputs: readonly Tensor[]): CompiledEntry | null {
    for (let i = 0; i < _cacheEntries.length; i++) {
      const entry = _cacheEntries[i];
      entry.shapeEnv.bindInputShapes(inputs);
      const { passed } = entry.shapeEnv.evaluateGuards();
      if (passed) return entry;
    }
    return null;
  }

  function _execute(entry: CompiledEntry, inputs: readonly Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    try {
      const out = executeCompiled(entry, inputs, entry.shapeEnv);
      if (_isThenable(out)) {
        return out.then(undefined, (e: unknown) => { throw _attachRepro(e, inputs, 'run'); });
      }
      return out;
    } catch (e) {
      throw _attachRepro(e, inputs, 'run');
    }
  }

  function compiledForward(...inputs: Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    let entry = _findCachedEntry(inputs);
    if (!entry) {
      const result = _compile(inputs);
      if (_isThenable(result)) {
        return result.then((c: CompiledEntry) => {
          _cacheEntries.push(c);
          c.shapeEnv.bindInputShapes(inputs);
          return _execute(c, inputs);
        });
      }
      entry = result;
      _cacheEntries.push(entry);
      entry.shapeEnv.bindInputShapes(inputs);
    }
    return _execute(entry, inputs);
  }

  let _ready: Promise<void> | null = null;
  if (exampleInputs) {
    const pending: MaybePromise<CompiledEntry>[] = [];
    if (shapeBuckets) {
      for (const bucketShapes of shapeBuckets) {
        pending.push(_compileWith(_bucketInputs(bucketShapes), null));
      }
    }
    pending.push(_compile(exampleInputs));

    if (pending.some(r => _isThenable(r))) {
      _ready = Promise.all(pending).then(entries => { for (const c of entries) _cacheEntries.push(c); });
    } else {
      for (const c of pending) {
        if (!_isThenable(c)) _cacheEntries.push(c);
      }
    }
  }

  const typedForward = compiledForward as CompiledForward;
  typedForward.original = model;

  typedForward.graph = (inputs?: Tensor[]) => {
    const args = inputs || exampleInputs;
    return trace(
      (...a: SymbolicTensor[]) => model.forward(...a),
      args!,
      { name: model.constructor.name || 'compiled', dynamicShapes }
    );
  };

  typedForward.source = () => {
    if (_cacheEntries.length === 0) return null;
    const compiled = _cacheEntries[0];
    const kernels = compiled.result.listKernels();
    return kernels.length > 0 ? compiled.result.getSource(kernels[0]) : null;
  };

  typedForward.kernels = () => {
    if (_cacheEntries.length === 0) return [];
    return _cacheEntries[0].result.listKernels();
  };

  typedForward.result = () => _cacheEntries.length > 0 ? _cacheEntries[0].result : null;
  typedForward._ready = _ready;

  return typedForward;
}
