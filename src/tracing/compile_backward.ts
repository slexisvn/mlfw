import { _traceCore, resolveDynamicShapes, inputSignatureOf, signatureMatches, isThenable } from './compile.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../backend/target.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { BackwardGraphBuilder } from '../compiler/ad/backward_builder.js';
import { IRBuilder } from '../compiler/ir/graph/builder.js';
import { JointGraphBuilder } from '../compiler/ad/joint_builder.js';
import { RematPolicy } from '../compiler/ad/remat_policy.js';
import type { InputSignature } from './types.js';
import '../compiler/ad/index.js';
import { tensorToContiguous, tensorToContiguousCopy, wrapResult } from '../dispatcher/jit_dispatch.js';
import { typedArrayCtor } from '../tensor/types/dtype.js';
import { computeNumel } from '../tensor/utils/shape_utils.js';
import { userArgIndexBounds } from '../compiler/analysis/index_bounds.js';
import { assertArgIndexBounds } from '../util/index_bounds.js';
import type { ArgIndexBound } from '../util/index_bounds.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { Device } from '../tensor/types/device.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { TensorType } from '../compiler/ir/graph/types.js';
import type { GraphFunction } from '../compiler/ir/graph/function.js';
import type { CompilableModel, CompiledResult, CompileOptions, GraphFunctionLike, GraphModuleLike, IRBuilderLike, IRValueLike, MaybePromise, RuntimeArg, SymbolicShape, TensorOutput, TracedCore } from './types.js';

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
  indexBounds: readonly ArgIndexBound[];
  inputSignature: InputSignature;
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
  indexBounds: readonly ArgIndexBound[];
  inputSignature: InputSignature;
};
type BackwardMeta = SeparateMeta | JointMeta;
type ArgSpec = { shape: number[]; dtype: DType };
type SeparateForwardContext = {
  results: TensorOutput | TensorOutput[];
  inputArrays: NumericTypedArray[];
  paramArrays: NumericTypedArray[];
  outputArrays: NumericTypedArray[];
  argSpecs: ArgSpec[];
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
  runJoint: (gradArrays: readonly NumericTypedArray[], outBufs: readonly NumericTypedArray[]) => AsyncKernelResult;
  pending: boolean;
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

export function compileWithBackward(model: CompilableModel, exampleInputs?: Tensor[], opts: CompileOptions = {}): unknown {
  const target = opts.target ?? CPUTarget();
  const mode = opts.mode || 'separate';
  const rematPolicy = opts.rematPolicy || new RematPolicy(opts.remat || {});
  const compilerOpts = { target, ...opts, backward: undefined, mode: undefined, rematPolicy: undefined, remat: undefined };
  const dynamicShapes = resolveDynamicShapes(opts);

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
      const indexBounds = userArgIndexBounds(t.graph as unknown as GraphModule, t.numUserInputs);
      const compiled = mode === 'joint'
        ? _compileJoint(func, t, rematPolicy)
        : _compileSeparate(func, t, rematPolicy);
      compiled.shapeEnv = t.shapeEnv;
      compiled.outputSymShapes = t.outputSymShapes;
      compiled.indexBounds = indexBounds;
      compiled.inputSignature = inputSignatureOf(inputs);
      return compiled;
    };

    return isThenable(traced) ? traced.then(finish) : finish(traced);
  }

  function _compileSeparate(forwardFunc: GraphFunctionLike, traced: TracedCore, policy: BackwardPolicy): SeparateMeta {
    const bwdBuilder = new BackwardGraphBuilder({ rematPolicy: policy as unknown as RematPolicy });
    const { backwardFunc, savedValues, gradInputIndices } = bwdBuilder.build(forwardFunc as unknown as GraphFunction) as unknown as FunctionBuilderResult;

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
      (new IRBuilder(forwardFunc as unknown as GraphFunction) as unknown as IRBuilderLike).returnOp([...realOutputs, ...extraSaved]);
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

    const fwdModule = new GraphModule('forward') as unknown as GraphModuleLike;
    fwdModule.addFunction(forwardFunc);
    const fwdResult = new Compiler(compilerOpts as never).compile(fwdModule as unknown as GraphModule) as unknown as CompiledResult;

    const bwdModule = new GraphModule('backward') as unknown as GraphModuleLike;
    bwdModule.addFunction(backwardFunc);
    const bwdResult = new Compiler(compilerOpts as never).compile(bwdModule as unknown as GraphModule) as unknown as CompiledResult;

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
      indexBounds: [],
      inputSignature: [],
    };
  }

  function _compileJoint(forwardFunc: GraphFunctionLike, traced: TracedCore, policy: BackwardPolicy): JointMeta {
    const jointBuilder = new JointGraphBuilder({ rematPolicy: policy as unknown as RematPolicy });
    const { jointFunc, numForwardOutputs, numGradInputs } = jointBuilder.build(forwardFunc as unknown as GraphFunction) as unknown as JointBuilderResult;

    const module = new GraphModule('joint') as unknown as GraphModuleLike;
    module.addFunction(jointFunc);
    const result = new Compiler(compilerOpts as never).compile(module as unknown as GraphModule) as unknown as CompiledResult;

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
      indexBounds: [],
      inputSignature: [],
    };
  }

  function _resolveOutputShape(compiled: BackwardMeta, i: number): number[] {
    if (compiled.outputSymShapes && compiled.shapeEnv) {
      return compiled.shapeEnv.resolveSymbolicShape(compiled.outputSymShapes[i]);
    }
    return [...compiled.outputTypes[i].shape as readonly number[]];
  }

  function _kernelIsAsync(result: CompiledResult, funcName: string): boolean {
    const mod = result.module || result;
    return !!mod.executionPlan || result.isAsync(funcName);
  }

  function _runK(result: CompiledResult, funcName: string, allArgs: RuntimeArg[]): AsyncKernelResult {
    if (!_kernelIsAsync(result, funcName)) {
      result.run(funcName, ...allArgs);
      return null;
    }
    const mod = result.module || result;
    return mod.executionPlan
      ? mod.runPlanAsync(mod.executionPlan, allArgs)
      : result.runAsync(funcName, ...allArgs);
  }

  function _allocOutputs(types: readonly TensorType[]): { arrays: NumericTypedArray[]; shapes: number[][] } {
    const arrays = new Array<NumericTypedArray>(types.length);
    const shapes = new Array<number[]>(types.length);
    for (let i = 0; i < types.length; i++) {
      const shape = [...types[i].shape as readonly number[]];
      const Ctor = typedArrayCtor(types[i].dtype);
      arrays[i] = new Ctor(Math.max(computeNumel(shape), 1));
      shapes[i] = shape;
    }
    return { arrays, shapes };
  }

  function _executeSeparateForward(compiled: SeparateMeta, inputs: readonly Tensor[]): MaybePromise<SeparateForwardContext> {
    const kernels = compiled.fwdResult.listKernels();
    const funcName = kernels[0];
    const device = (inputs.length > 0 ? inputs[0].device : 'cpu') as Device;

    const inputArrays = inputs.map((t: Tensor) => tensorToContiguousCopy(t));
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

    const argSpecs: ArgSpec[] = [...inputs, ...params].map((t: Tensor) => ({ shape: [...t.shape], dtype: t.dtype }));

    const allArgs: RuntimeArg[] = [...inputArrays, ...paramArrays, ...outputArrays];
    const build = () => {
      const results = numRealOutputs === 1
        ? wrapResult(outputArrays[0], outputShapes[0], allOutputTypes[0].dtype, device)
        : Array.from({ length: numRealOutputs }, (_, i) => wrapResult(outputArrays[i], outputShapes[i], allOutputTypes[i].dtype, device));
      return { results, inputArrays, paramArrays, outputArrays, argSpecs, device };
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
    const build = () => {
      const gradInputs = new Array<TensorOutput>(savedContext.argSpecs.length);
      for (let i = 0; i < numGradOutputs; i++) {
        gradInputs[compiled.gradInputIndices[i]] =
          wrapResult(gradInputArrays[i], gradInputShapes[i], bwdFunc.outputTypes[i].dtype, savedContext.device);
      }
      for (let i = 0; i < gradInputs.length; i++) {
        if (gradInputs[i]) continue;
        const spec = savedContext.argSpecs[i];
        const Ctor = typedArrayCtor(spec.dtype);
        gradInputs[i] = wrapResult(new Ctor(Math.max(computeNumel(spec.shape), 1)), spec.shape, spec.dtype, savedContext.device);
      }
      return gradInputs;
    };
    const pending = _runK(compiled.bwdResult, funcName, allArgs);
    return pending ? pending.then(build) : build();
  }

  function _findCachedEntry(inputs: readonly Tensor[]): BackwardMeta | null {
    const signature = inputSignatureOf(inputs);
    for (let i = 0; i < _cacheEntries.length; i++) {
      const entry = _cacheEntries[i];
      if (!signatureMatches(entry.inputSignature, signature)) continue;
      entry.shapeEnv.bindInputShapes(inputs);
      const { passed } = entry.shapeEnv.evaluateGuards();
      if (passed) return entry;
    }
    return null;
  }

  function _forwardWith(meta: BackwardMeta, inputs: readonly Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    assertArgIndexBounds(meta.indexBounds, inputs);
    _activeMeta = meta;
    if (meta.mode === 'joint') {
      return _executeJointForward(meta, inputs);
    }
    const ctx = _executeSeparateForward(meta, inputs);
    if (isThenable(ctx)) return ctx.then((c: SeparateForwardContext) => { _savedValues = c; return c.results; });
    _savedValues = ctx;
    return ctx.results;
  }

  function _runForward(inputs: Tensor[]): MaybePromise<TensorOutput | TensorOutput[]> {
    const cached = _findCachedEntry(inputs);
    if (cached) return _forwardWith(cached, inputs);

    const compiledOrPromise = _compile(inputs);
    if (isThenable(compiledOrPromise)) {
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

    const inputArrays = inputs.map((t: Tensor) => tensorToContiguousCopy(t));
    const params = compiled.capturedParams;
    const paramArrays = params.map((t: Tensor) => tensorToContiguous(t));

    const jointFunc = compiled.jointFunc;
    const { arrays: outputArrays, shapes: outputShapes } = _allocOutputs(jointFunc.outputTypes);
    const gradOutputArrays = _allocOutputs(compiled.outputTypes.slice(0, compiled.numForwardOutputs)).arrays;

    const ctx: JointSavedContext = {
      inputArrays, paramArrays, gradOutputArrays, outputArrays, outputShapes, device, compiled,
      runJoint: (gradArrays, outBufs) => _runK(
        compiled.result, funcName,
        [...inputArrays, ...paramArrays, ...gradArrays, ...outBufs],
      ),
      pending: !_kernelIsAsync(compiled.result, funcName),
    };
    _savedValues = ctx;

    const build = () => {
      const fwdOutputs = [];
      for (let i = 0; i < compiled.numForwardOutputs; i++) {
        const out = wrapResult(outputArrays[i], outputShapes[i], jointFunc.outputTypes[i].dtype, device);
        out.storage.setPendingFill(() => { if (ctx.pending) _settleJoint(ctx, ctx.gradOutputArrays, outputArrays); });
        fwdOutputs.push(out);
      }
      return fwdOutputs.length === 1 ? fwdOutputs[0] : fwdOutputs;
    };

    if (ctx.pending) return build();
    const pending = ctx.runJoint(gradOutputArrays, outputArrays);
    return pending ? pending.then(build) : build();
  }

  function _settleJoint(ctx: JointSavedContext, gradArrays: readonly NumericTypedArray[], outBufs: readonly NumericTypedArray[]): AsyncKernelResult {
    ctx.pending = false;
    return ctx.runJoint(gradArrays, outBufs);
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

    const reference = _activeMeta.mode === 'joint'
      ? (_savedValues as JointSavedContext).gradOutputArrays
      : (_savedValues as SeparateForwardContext).outputArrays.slice(0, _activeMeta.numRealOutputs);
    if (gradOutputs.length !== reference.length) {
      throw new Error(
        `backward() expects one gradient per forward output: the model returns ${reference.length} output(s) but ${gradOutputs.length} gradient(s) were given`
      );
    }
    for (let i = 0; i < gradOutputs.length; i++) {
      const expected = reference[i].length;
      const got = computeNumel(gradOutputs[i].shape);
      if (got !== expected) {
        throw new Error(
          `backward() gradient ${i} has ${got} element(s) but forward output ${i} has ${expected}`
        );
      }
    }

    if (_activeMeta.mode === 'joint') {
      return _executeJointBackward(_activeMeta, gradOutputs, _savedValues as JointSavedContext);
    }

    return _executeSeparateBackward(_activeMeta, gradOutputs, _savedValues as SeparateForwardContext);
  };

  function _executeJointBackward(compiled: JointMeta, gradOutputs: readonly Tensor[], savedCtx: JointSavedContext): MaybePromise<TensorOutput[]> {
    const { outputArrays, device } = savedCtx;
    const gradArrays = gradOutputs.map((t: Tensor) => tensorToContiguous(t));

    const jointFunc = compiled.jointFunc;
    const gradInputTypes = jointFunc.outputTypes.slice(compiled.numForwardOutputs);
    const { arrays: gradInputArrays, shapes: gradInputShapes } = _allocOutputs(gradInputTypes);
    const outBufs = [...outputArrays.slice(0, compiled.numForwardOutputs), ...gradInputArrays];

    const build = () => gradInputArrays.map(
      (arr, i) => wrapResult(arr, gradInputShapes[i], gradInputTypes[i].dtype, device)
    );
    const pending = _settleJoint(savedCtx, gradArrays, outBufs);
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
    if (isThenable(compiled)) {
      _pendingCompile = compiled.then((meta) => { _cacheEntries.push(meta); _pendingCompile = null; return meta; });
    } else {
      _cacheEntries.push(compiled);
    }
  }

  return typedForward;
}
