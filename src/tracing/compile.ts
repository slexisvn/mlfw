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

import { foldWeightParams, weightPredicate, MAX_FOLDABLE_ELEMENTS } from './fold_params.js';
import { compileWithBackward } from './compile_backward.js';
import { userArgIndexBounds } from '../compiler/analysis/index_bounds.js';
import { assertArgIndexBounds } from '../util/index_bounds.js';
import { cloneGraphModule } from '../compiler/ir/graph/module.js';
import {
  BASELINE, candidateByName, gateCacheKey, graphSignature, optimizationCandidates, selectWinner,
} from '../compiler/pipeline/opt_gate.js';
import type { ArgIndexBound } from '../util/index_bounds.js';
import type { CandidateMeasurement, GateDecision, GateTarget } from '../compiler/pipeline/opt_gate.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { Device } from '../tensor/types/device.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { TensorType } from '../compiler/ir/graph/types.js';
import type { ShapeEnv } from './shape_env.js';
import type { SymbolicTensor } from './symbolic_tensor.js';
import type { CompilableModel, CompiledEntry, CompiledForward, CompiledResult, CompileOptions, DynamicShapes, GraphModuleLike, MaybePromise, RuntimeArg, RuntimeTensorLike, SymbolicShape, TensorOutput, TracedCore } from './types.js';
import type { GraphModule } from '../compiler/ir/graph/module.js';

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

function sourceForEntry(compiled: CompiledEntry, entryIndex: number): string | null {
  const chunks: string[] = [];
  for (const kernel of compiled.result.listKernels()) {
    const source = compiled.result.getSource(kernel);
    if (source === null) continue;
    chunks.push(`// ---- compiled entry ${entryIndex}, kernel ${kernel} ----\n${source}`);
  }
  return chunks.length > 0 ? chunks.join('\n\n') : null;
}

const _gateDecisions = new Map<string, string>();

const GATE_WARMUP = 2;
const GATE_REPEAT = 7;
const GATE_TOL = 2e-3;

export function clearOptimizationGateCache(): void {
  _gateDecisions.clear();
}

function _flatOf(out: unknown): number[] {
  const items = Array.isArray(out) ? out : [out];
  const values: number[] = [];
  for (const t of items) {
    const data = (t as { contiguous?: () => { data: ArrayLike<number> }; data?: ArrayLike<number> });
    const arr = data.contiguous ? data.contiguous().data : data.data;
    if (!arr) continue;
    for (let i = 0; i < arr.length; i++) values.push(Number(arr[i]));
  }
  return values;
}

function _closeEnough(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(b[i])) return false;
    if (Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])) > GATE_TOL) return false;
  }
  return true;
}

function _timeEntry(entry: CompiledEntry, inputs: readonly Tensor[]): { ms: number; values: number[] } | null {
  entry.shapeEnv.bindInputShapes(inputs);
  let last: unknown;
  for (let i = 0; i < GATE_WARMUP; i++) {
    last = executeCompiled(entry, inputs, entry.shapeEnv);
    if (_isThenable(last)) return null;
  }
  const samples: number[] = [];
  for (let i = 0; i < GATE_REPEAT; i++) {
    const t0 = performance.now();
    last = executeCompiled(entry, inputs, entry.shapeEnv);
    samples.push(performance.now() - t0);
  }
  samples.sort((x, y) => x - y);
  return { ms: samples[Math.floor(samples.length / 2)], values: _flatOf(last) };
}

export function compile(model: CompilableModel, exampleInputs?: Tensor[], opts: CompileOptions = {}): unknown {
  if (opts?.backward) {
    return compileWithBackward(model, exampleInputs, opts);
  }

  const target = opts?.target ?? CPUTarget();
  const compilerOpts: Record<string, unknown> & { target: unknown } = { target, ...opts };
  const dynamicShapes = resolveDynamicShapes(opts);
  const shapeBuckets = opts?.shapeBuckets || null;
  const foldWeights = opts?.foldWeights ?? opts?.quantization?.foldWeights ?? false;
  const tuneOptimizations = opts?.tuneOptimizations ?? false;

  const _cacheEntries: CompiledEntry[] = [];
  const _gateReports: GateDecision[] = [];

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

  function _bindCalibrationParams(base: Record<string, unknown>, params: readonly Tensor[]): Record<string, unknown> {
    const q = base.quantization as { calibrationData?: readonly unknown[] } | undefined;
    if (!q || !q.calibrationData) return base;
    const asArray = (v: unknown) => (ArrayBuffer.isView(v) ? v : tensorToContiguous(v as Tensor));
    const paramArrays = params.map(asArray);
    const calibrationData = q.calibrationData.map((batch) => {
      const given = Array.isArray(batch) ? batch : [batch];
      return [...given.map(asArray), ...paramArrays];
    });
    return { ...base, quantization: { ...q, calibrationData } };
  }

  const quantizing = !!(opts?.quantization as { enabled?: boolean } | undefined)?.enabled;
  const linksConstants = quantizing || !!(target as { supportsConstBuffers?: boolean })?.supportsConstBuffers;
  const foldPredicate = weightPredicate(linksConstants ? Infinity : MAX_FOLDABLE_ELEMENTS);

  function _entryFor(prepared: TracedCore, indexBounds: readonly ArgIndexBound[], overrides: Readonly<Record<string, unknown>> | null): CompiledEntry {
    const opts = overrides
      ? {
        ...compilerOpts as Record<string, unknown>,
        optimization: { ...(compilerOpts as { optimization?: object }).optimization, ...(overrides.optimization as object ?? {}) },
        scheduling: { ...(compilerOpts as { scheduling?: object }).scheduling, ...(overrides.scheduling as object ?? {}) },
      }
      : compilerOpts as Record<string, unknown>;
    const module = overrides
      ? cloneGraphModule(prepared.graph as unknown as GraphModule)
      : prepared.graph as unknown as GraphModule;
    const result = new Compiler(_bindCalibrationParams(opts, prepared.capturedParams) as never).compile(module) as unknown as CompiledResult;
    return {
      result,
      graph: module as unknown as GraphModuleLike,
      capturedParams: prepared.capturedParams,
      numUserInputs: prepared.numUserInputs,
      outputTypes: prepared.outputTypes,
      shapeEnv: prepared.shapeEnv,
      outputSymShapes: prepared.outputSymShapes,
      indexBounds,
    };
  }

  function _finalize(traced: TracedCore): CompiledEntry {
    const prepared = foldWeights ? foldWeightParams(traced, tensorToContiguous, foldPredicate) : traced;
    const indexBounds = userArgIndexBounds(prepared.graph as unknown as GraphModule, prepared.numUserInputs);
    const tuned = tuneOptimizations
      ? runOptimizationGate(prepared, indexBounds, _entryFor)
      : null;
    return tuned ?? _entryFor(prepared, indexBounds, null);
  }

  function runOptimizationGate(
    prepared: TracedCore,
    indexBounds: readonly ArgIndexBound[],
    build: (p: TracedCore, b: readonly ArgIndexBound[], o: Readonly<Record<string, unknown>> | null) => CompiledEntry,
  ): CompiledEntry | null {
    const gateTarget = target as unknown as GateTarget & { name?: string };
    const candidates = optimizationCandidates(gateTarget);
    if (candidates.length === 0 || !exampleInputs) return null;

    const func = (prepared.graph as unknown as GraphModule).functions().next().value;
    const opNames: string[] = [];
    if (func) for (const op of func.ops()) opNames.push(op.opName);
    const key = gateCacheKey(
      graphSignature(opNames, exampleInputs.map(t => t.shape)),
      gateTarget?.name ?? 'unknown',
      candidates,
    );

    const cached = _gateDecisions.get(key);
    if (cached !== undefined) {
      return build(prepared, indexBounds, candidateByName(candidates, cached)?.optimization ? { optimization: candidateByName(candidates, cached)!.optimization } : null);
    }

    const measurements: CandidateMeasurement[] = [];
    let baselineEntry: CompiledEntry | null = null;
    let reference: number[] | null = null;
    const winners = new Map<string, CompiledEntry>();

    for (const spec of [{ name: BASELINE, optimization: undefined }, ...candidates]) {
      let entry: CompiledEntry;
      try {
        entry = build(prepared, indexBounds, spec.name === BASELINE ? null : { optimization: spec.optimization });
      } catch (e) {
        measurements.push({ name: spec.name, ms: 0, correct: false, error: String((e as Error)?.message ?? e) });
        continue;
      }
      const timed = _timeEntry(entry, exampleInputs!);
      if (!timed) {
        measurements.push({ name: spec.name, ms: 0, correct: false, error: 'runtime is asynchronous; the gate only measures synchronous runtimes' });
        if (spec.name === BASELINE) return null;
        continue;
      }
      if (spec.name === BASELINE) {
        reference = timed.values;
        baselineEntry = entry;
      }
      const correct = reference !== null && _closeEnough(reference, timed.values);
      measurements.push({ name: spec.name, ms: timed.ms, correct });
      if (correct) winners.set(spec.name, entry);
    }

    if (!baselineEntry) return null;
    const decision = selectWinner(measurements);
    _gateDecisions.set(key, decision.winner);
    _gateReports.push(decision);
    return winners.get(decision.winner) ?? baselineEntry;
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
    assertArgIndexBounds(entry.indexBounds, inputs);
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
    for (let i = 0; i < inputs.length; i++) {
      if (!inputs[i] || !Array.isArray(inputs[i].shape)) {
        throw new Error(`compiled model: argument ${i} is not a Tensor (received ${inputs[i] === undefined ? 'undefined' : typeof inputs[i]})`);
      }
    }
    if (exampleInputs && inputs.length !== exampleInputs.length) {
      throw new Error(`compiled model: expected ${exampleInputs.length} input tensor(s) but got ${inputs.length}`);
    }
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
  typedForward.tuningReport = () => [..._gateReports];
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
    return sourceForEntry(_cacheEntries[0], 0);
  };

  typedForward.kernels = () => {
    if (_cacheEntries.length === 0) return [];
    return _cacheEntries[0].result.listKernels();
  };

  typedForward.result = () => _cacheEntries.length > 0 ? _cacheEntries[0].result : null;
  typedForward._ready = _ready;

  return typedForward;
}
