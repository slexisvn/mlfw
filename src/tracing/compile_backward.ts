import { _traceCore } from './compile.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../backend/target.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { BackwardGraphBuilder } from '../compiler/ad/backward_builder.js';
import { IRBuilder } from '../compiler/ir/graph/builder.js';
import { JointGraphBuilder } from '../compiler/ad/joint_builder.js';
import { RematPolicy } from '../compiler/ad/remat_policy.js';
import '../compiler/ad/index.js';
import { tensorToContiguous, wrapResult } from '../dispatcher/jit_dispatch.js';
import { typedArrayCtor } from '../tensor/types/dtype.js';
import { computeNumel } from '../tensor/utils/shape_utils.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { Device } from '../tensor/types/device.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';
import type { TensorType } from '../compiler/ir/graph/types.js';
import type { CompilableModel, CompiledResult, CompileOptions, GraphFunctionLike, GraphModuleLike, IRBuilderLike, IROperationLike, IRValueLike, MaybePromise, RuntimeArg, SymbolicShape, TensorOutput, TracedCore } from './types.js';

type SavedSource = { kind: 'arg' | 'output'; index: number };
type BackwardPolicy = unknown;
type SeparateMeta = {
  mode: 'separate';
  fwdResult: CompiledResult;
  bwdResult: CompiledResult;
  forwardFunc: GraphFunctionLike;
  backwardFunc: GraphFunctionLike;
  savedValues: IRValueLike[];
  savedSources: SavedSource[];
  numRealOutputs: number;
  gradInputIndices: number[];
  capturedParams: Tensor[];
  numUserInputs: number;
  outputTypes: readonly TensorType[];
  shapeEnv: TracedCore['shapeEnv'];
  outputSymShapes: readonly SymbolicShape[];
};
type JointMeta = {
  mode: 'joint';
  result: CompiledResult;
  jointFunc: GraphFunctionLike;
  numForwardOutputs: number;
  numGradInputs: number;
  capturedParams: Tensor[];
  numUserInputs: number;
  outputTypes: readonly TensorType[];
  inputTypes: readonly TensorType[];
  shapeEnv: TracedCore['shapeEnv'];
  outputSymShapes: readonly SymbolicShape[];
};
type BackwardMeta = SeparateMeta | JointMeta;
type SeparateForwardContext = {
  results: TensorOutput | TensorOutput[];
  inputArrays: NumericTypedArray[];
  paramArrays: NumericTypedArray[];
  outputArrays: NumericTypedArray[];
  device: Device;
};
type JointSavedContext = {
  inputArrays: NumericTypedArray[];
  paramArrays: NumericTypedArray[];
  gradOutputArrays: NumericTypedArray[];
  outputArrays: NumericTypedArray[];
  outputShapes: number[][];
  device: Device;
  compiled: JointMeta;
};
type SavedContext = SeparateForwardContext | JointSavedContext;
type FunctionBuilderResult = {
  backwardFunc: GraphFunctionLike;
  savedValues: IRValueLike[];
  gradInputIndices: number[];
};
type JointBuilderResult = {
  jointFunc: GraphFunctionLike;
  numForwardOutputs: number;
  numGradInputs: number;
};
type AsyncKernelResult = Promise<unknown> | null;

function _isThenable<T>(value: MaybePromise<T>): value is Promise<T> {
  return typeof (value as { then?: unknown }).then === 'function';
}

export function compileWithBackward(model: CompilableModel, exampleInputs?: Tensor[], opts: CompileOptions = {}): unknown {
  const target = opts.target ?? CPUTarget();
  const mode = opts.mode || 'separate';
  const rematPolicy = opts.rematPolicy || new RematPolicy(opts.remat || {});
  const compilerOpts = { target, ...opts, backward: undefined, mode: undefined, rematPolicy: undefined, remat: undefined };
  const dynamicShapes = opts.dynamic_shapes || null;

  const _cacheEntries: BackwardMeta[] = [];
  let _savedValues: SavedContext | null = null;
  let _activeMeta: BackwardMeta | null = null;
  let _pendingCompile: Promise<BackwardMeta> | null = null;

  function _compile(inputs: Tensor[]): MaybePromise<BackwardMeta> {
    const traced = _traceCore(
      (...args: TensorOutput[]) => model.forward(...args),
      inputs,
      { name: model.constructor.name || 'compiled', dynamicShapes }
    );

    const finish = (t: TracedCore): BackwardMeta => {
      const func = t.graph.functions().next().value as GraphFunctionLike;
      const compiled = mode === 'joint'
        ? _compileJoint(func, t, rematPolicy)
        : _compileSeparate(func, t, rematPolicy);
      compiled.shapeEnv = t.shapeEnv;
      compiled.outputSymShapes = t.outputSymShapes;
      return compiled;
    };

    return _isThenable(traced) ? traced.then(finish) : finish(traced);
  }

  function _compileSeparate(forwardFunc: GraphFunctionLike, traced: TracedCore, policy: BackwardPolicy): SeparateMeta {
    const bwdBuilder = new BackwardGraphBuilder({ rematPolicy: policy });
    const { backwardFunc, savedValues, gradInputIndices } = bwdBuilder.build(forwardFunc) as FunctionBuilderResult;

    const realReturnOp = forwardFunc.getReturnOp();
    const realOutputs = [...realReturnOp!.operands];
    const numRealOutputs = realOutputs.length;

    const argIndexById = new Map(forwardFunc.args.map((a: IRValueLike, i: number) => [a.id, i]));
    const outputIndexById = new Map(realOutputs.map((o: IRValueLike, i: number) => [o.id, i]));

    const extraSaved: IRValueLike[] = [];
    const extraIndexById = new Map<number | undefined, number>();
    for (const sv of savedValues) {
      if (argIndexById.has(sv.id) || outputIndexById.has(sv.id) || extraIndexById.has(sv.id)) continue;
      extraIndexById.set(sv.id, numRealOutputs + extraSaved.length);
      extraSaved.push(sv);
    }

    if (extraSaved.length > 0) {
      realReturnOp!.erase();
      (new IRBuilder(forwardFunc) as IRBuilderLike).returnOp([...realOutputs, ...extraSaved]);
      forwardFunc.outputTypes = Object.freeze([
        ...realOutputs.map(v => v.type),
        ...extraSaved.map(v => v.type),
      ]);
    }

    const savedSources: SavedSource[] = savedValues.map((sv: IRValueLike) => {
      if (argIndexById.has(sv.id)) return { kind: 'arg', index: argIndexById.get(sv.id) };
      if (outputIndexById.has(sv.id)) return { kind: 'output', index: outputIndexById.get(sv.id) };
      return { kind: 'output', index: extraIndexById.get(sv.id) };
    }) as SavedSource[];

    const fwdModule = new GraphModule('forward') as GraphModuleLike;
    fwdModule.addFunction(forwardFunc);
    const fwdResult = new Compiler(compilerOpts).compile(fwdModule) as CompiledResult;

    const bwdModule = new GraphModule('backward') as GraphModuleLike;
    bwdModule.addFunction(backwardFunc);
    const bwdResult = new Compiler(compilerOpts).compile(bwdModule) as CompiledResult;

    return {
      mode: 'separate',
      fwdResult,
      bwdResult,
      forwardFunc,
      backwardFunc,
      savedValues,
      savedSources,
      numRealOutputs,
      gradInputIndices,
      capturedParams: traced.capturedParams,
      numUserInputs: traced.numUserInputs,
      outputTypes: traced.outputTypes,
      shapeEnv: traced.shapeEnv,
      outputSymShapes: traced.outputSymShapes,
    };
  }

  function _compileJoint(forwardFunc: GraphFunctionLike, traced: TracedCore, policy: BackwardPolicy): JointMeta {
    const jointBuilder = new JointGraphBuilder({ rematPolicy: policy });
    const { jointFunc, numForwardOutputs, numGradInputs } = jointBuilder.build(forwardFunc) as JointBuilderResult;

    const module = new GraphModule('joint') as GraphModuleLike;
    module.addFunction(jointFunc);
    const result = new Compiler(compilerOpts).compile(module) as CompiledResult;

    return {
      mode: 'joint',
      result,
      jointFunc,
      numForwardOutputs,
      numGradInputs,
      capturedParams: traced.capturedParams,
      numUserInputs: traced.numUserInputs,
      outputTypes: traced.outputTypes,
      inputTypes: forwardFunc.inputTypes,
      shapeEnv: traced.shapeEnv,
      outputSymShapes: traced.outputSymShapes,
    };
  }

  function _resolveOutputShape(compiled: BackwardMeta, i: number): number[] {
    if (compiled.outputSymShapes && compiled.shapeEnv) {
      return compiled.shapeEnv.resolveSymbolicShape(compiled.outputSymShapes[i]);
    }
    return [...compiled.outputTypes[i].shape as readonly number[]];
  }

  function _runK(result: CompiledResult, funcName: string, allArgs: RuntimeArg[]): AsyncKernelResult {
    const mod = result.module || result;
    if (mod.executionPlan) return mod.runPlanAsync(mod.executionPlan, allArgs);
    if (result.isAsync(funcName)) return result.runAsync(funcName, ...allArgs);
    result.run(funcName, ...allArgs);
    return null;
  }

  function _executeSeparateForward(compiled: SeparateMeta, inputs: readonly Tensor[]): MaybePromise<SeparateForwardContext> {
    const kernels = compiled.fwdResult.listKernels();
    const funcName = kernels[0];
    const device = (inputs.length > 0 ? inputs[0].device : 'cpu') as Device;

    const inputArrays = inputs.map((t: Tensor) => tensorToContiguous(t));
    const params = compiled.capturedParams;
    const paramArrays = params.map((t: Tensor) => tensorToContiguous(t));

    const allOutputTypes = compiled.forwardFunc.outputTypes;
    const numRealOutputs = compiled.numRealOutputs;
    const outputArrays = new Array<NumericTypedArray>(allOutputTypes.length);
    const outputShapes = new Array<number[]>(allOutputTypes.length);
    for (let i = 0; i < allOutputTypes.length; i++) {
      const shape = i < numRealOutputs ? _resolveOutputShape(compiled, i) : [...allOutputTypes[i].shape as readonly number[]];
      const dtype = allOutputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      outputArrays[i] = new Ctor(Math.max(numel, 1));
      outputShapes[i] = shape;
    }

    const allArgs: RuntimeArg[] = [...inputArrays, ...paramArrays, ...outputArrays];
    const build = () => {
      const results = numRealOutputs === 1
        ? wrapResult(outputArrays[0], outputShapes[0], allOutputTypes[0].dtype, device)
        : Array.from({ length: numRealOutputs }, (_, i) => wrapResult(outputArrays[i], outputShapes[i], allOutputTypes[i].dtype, device));
      return { results, inputArrays, paramArrays, outputArrays, device };
    };
    const pending = _runK(compiled.fwdResult, funcName, allArgs);
    return pending ? pending.then(build) : build();
  }

  function _executeSeparateBackward(compiled: SeparateMeta, gradOutputs: readonly Tensor[], savedContext: SeparateForwardContext): MaybePromise<TensorOutput[]> {
    const kernels = compiled.bwdResult.listKernels();
    const funcName = kernels[0];

    const gradArrays = gradOutputs.map((t: Tensor) => tensorToContiguous(t));

    const savedValues = compiled.savedValues;
    const savedSources = compiled.savedSources;
    const argBuffers = [...savedContext.inputArrays, ...savedContext.paramArrays];
    const savedArrays = new Array(savedValues.length);
    for (let i = 0; i < savedValues.length; i++) {
      const src = savedSources[i];
      savedArrays[i] = src.kind === 'arg' ? argBuffers[src.index] : savedContext.outputArrays[src.index];
    }

    const bwdFunc = compiled.backwardFunc;
    const numGradOutputs = bwdFunc.outputTypes.length;
    const gradInputArrays = new Array<NumericTypedArray>(numGradOutputs);
    const gradInputShapes = new Array<number[]>(numGradOutputs);
    for (let i = 0; i < numGradOutputs; i++) {
      const shape = [...bwdFunc.outputTypes[i].shape as readonly number[]];
      const dtype = bwdFunc.outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      gradInputArrays[i] = new Ctor(Math.max(numel, 1));
      gradInputShapes[i] = shape;
    }

    const allArgs: RuntimeArg[] = [...gradArrays, ...savedArrays, ...gradInputArrays];
    const build = () => gradInputArrays.map((arr, i) =>
      wrapResult(arr, gradInputShapes[i], bwdFunc.outputTypes[i].dtype, savedContext.device)
    );
    const pending = _runK(compiled.bwdResult, funcName, allArgs);
    return pending ? pending.then(build) : build();
  }

  function _findCachedEntry(inputs: readonly Tensor[]): BackwardMeta | null {
    for (let i = 0; i < _cacheEntries.length; i++) {
      const entry = _cacheEntries[i];
      entry.shapeEnv.bindInputShapes(inputs);
      const { passed } = entry.shapeEnv.evaluateGuards();
      if (passed) return entry;
    }
    return null;
  }

  function _forwardWith(meta: BackwardMeta, inputs: readonly Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    _activeMeta = meta;
    if (meta.mode === 'joint') {
      return _executeJointForward(meta, inputs);
    }
    const ctx = _executeSeparateForward(meta, inputs);
    if (_isThenable(ctx)) return ctx.then((c: SeparateForwardContext) => { _savedValues = c; return c.results; });
    _savedValues = ctx;
    return ctx.results;
  }

  function _runForward(inputs: Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    const cached = _findCachedEntry(inputs);
    if (cached) return _forwardWith(cached, inputs);

    const compiledOrPromise = _compile(inputs);
    if (_isThenable(compiledOrPromise)) {
      return compiledOrPromise.then((meta) => {
        _cacheEntries.push(meta);
        meta.shapeEnv.bindInputShapes(inputs);
        return _forwardWith(meta, inputs);
      });
    }
    _cacheEntries.push(compiledOrPromise);
    compiledOrPromise.shapeEnv.bindInputShapes(inputs);
    return _forwardWith(compiledOrPromise, inputs);
  }

  function compiledForward(...inputs: Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    if (_pendingCompile) return _pendingCompile.then(() => _runForward(inputs));
    return _runForward(inputs);
  }

  function _executeJointForward(compiled: JointMeta, inputs: readonly Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    const kernels = compiled.result.listKernels();
    const funcName = kernels[0];
    const device = (inputs.length > 0 ? inputs[0].device : 'cpu') as Device;

    const inputArrays = inputs.map((t: Tensor) => tensorToContiguous(t));
    const params = compiled.capturedParams;
    const paramArrays = params.map((t: Tensor) => tensorToContiguous(t));

    const jointFunc = compiled.jointFunc;
    const numOutputs = jointFunc.outputTypes.length;
    const outputArrays = new Array<NumericTypedArray>(numOutputs);
    const outputShapes = new Array<number[]>(numOutputs);
    for (let i = 0; i < numOutputs; i++) {
      const shape = [...jointFunc.outputTypes[i].shape as readonly number[]];
      const dtype = jointFunc.outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      outputArrays[i] = new Ctor(Math.max(numel, 1));
      outputShapes[i] = shape;
    }

    const gradOutputArrays = new Array<NumericTypedArray>(compiled.numForwardOutputs);
    for (let i = 0; i < compiled.numForwardOutputs; i++) {
      const type = compiled.outputTypes[i];
      const numel = computeNumel(type.shape as readonly number[]);
      const Ctor = typedArrayCtor(type.dtype);
      gradOutputArrays[i] = new Ctor(Math.max(numel, 1));
    }

    _savedValues = { inputArrays, paramArrays, gradOutputArrays, outputArrays, outputShapes, device, compiled };

    const allArgs: RuntimeArg[] = [...inputArrays, ...paramArrays, ...gradOutputArrays, ...outputArrays];
    const build = () => {
      const fwdOutputs = [];
      for (let i = 0; i < compiled.numForwardOutputs; i++) {
        fwdOutputs.push(wrapResult(outputArrays[i], outputShapes[i], jointFunc.outputTypes[i].dtype, device));
      }
      return fwdOutputs.length === 1 ? fwdOutputs[0] : fwdOutputs;
    };
    const pending = _runK(compiled.result, funcName, allArgs);
    return pending ? pending.then(build) : build();
  }

  const typedForward = compiledForward as typeof compiledForward & {
    backward(...gradOutputs: Tensor[]): MaybePromise<TensorOutput[]>;
    original: CompilableModel;
    backwardGraph(): GraphFunctionLike | null;
    forwardGraph(): GraphFunctionLike | null;
    capturedParams(): Tensor[];
  };

  typedForward.backward = function (...gradOutputs: Tensor[]): MaybePromise<TensorOutput[]> {
    if (!_activeMeta || !_savedValues) {
      throw new Error('Must run forward before backward');
    }

    if (_activeMeta.mode === 'joint') {
      return _executeJointBackward(_activeMeta, gradOutputs, _savedValues as JointSavedContext);
    }

    return _executeSeparateBackward(_activeMeta, gradOutputs, _savedValues as SeparateForwardContext);
  };

  function _executeJointBackward(compiled: JointMeta, gradOutputs: readonly Tensor[], savedCtx: JointSavedContext): MaybePromise<TensorOutput[]> {
    const { inputArrays, paramArrays, outputArrays, outputShapes, device } = savedCtx;

    const gradArrays = gradOutputs.map((t: Tensor) => tensorToContiguous(t));
    for (let i = 0; i < gradArrays.length; i++) {
      (savedCtx.gradOutputArrays[i] as { set(values: unknown): void }).set(gradArrays[i]);
    }

    const jointFunc = compiled.jointFunc;
    const numOutputs = jointFunc.outputTypes.length;
    const newOutputArrays = new Array<NumericTypedArray>(numOutputs);
    const newOutputShapes = new Array<number[]>(numOutputs);
    for (let i = 0; i < numOutputs; i++) {
      const shape = [...jointFunc.outputTypes[i].shape as readonly number[]];
      const dtype = jointFunc.outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      newOutputArrays[i] = new Ctor(Math.max(numel, 1));
      newOutputShapes[i] = shape;
    }

    const allArgs: RuntimeArg[] = [...inputArrays, ...paramArrays, ...gradArrays, ...newOutputArrays];
    const build = () => {
      const gradInputs = [];
      for (let i = compiled.numForwardOutputs; i < numOutputs; i++) {
        gradInputs.push(wrapResult(
          newOutputArrays[i],
          newOutputShapes[i],
          jointFunc.outputTypes[i].dtype,
          device
        ));
      }
      return gradInputs;
    };
    const pending = _runK(compiled.result, compiled.result.listKernels()[0], allArgs);
    return pending ? pending.then(build) : build();
  }

  typedForward.original = model;

  typedForward.backwardGraph = () => {
    if (_cacheEntries.length === 0) return null;
    const meta = _cacheEntries[0];
    if (meta.mode === 'joint') return meta.jointFunc;
    return meta.backwardFunc;
  };

  typedForward.forwardGraph = () => {
    if (_cacheEntries.length === 0) return null;
    const meta = _cacheEntries[0];
    if (meta.mode === 'joint') return meta.jointFunc;
    return meta.forwardFunc;
  };

  typedForward.capturedParams = () => (_cacheEntries.length ? _cacheEntries[0].capturedParams : []);

  if (exampleInputs) {
    const compiled = _compile(exampleInputs);
    if (_isThenable(compiled)) {
      _pendingCompile = compiled.then((meta) => { _cacheEntries.push(meta); _pendingCompile = null; return meta; });
    } else {
      _cacheEntries.push(compiled);
    }
  }

  return typedForward;
}
