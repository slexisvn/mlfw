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

export function compileWithBackward(model, exampleInputs, opts = {}) {
  const target = opts.target ?? CPUTarget();
  const mode = opts.mode || 'separate';
  const rematPolicy = opts.rematPolicy || new RematPolicy(opts.remat || {});
  const compilerOpts = { target, verify: false, ...opts, backward: undefined, mode: undefined, rematPolicy: undefined, remat: undefined };
  const dynamicShapes = opts.dynamic_shapes || null;

  const _cacheEntries = [];
  let _savedValues = null;
  let _activeMeta = null;

  function _compile(inputs) {
    const traced = _traceCore(
      (...args) => model.forward(...args),
      inputs,
      { name: model.constructor.name || 'compiled', dynamicShapes }
    );

    const graph = traced.graph;
    const func = graph.functions().next().value;

    const compiled = mode === 'joint'
      ? _compileJoint(func, traced, rematPolicy)
      : _compileSeparate(func, traced, rematPolicy);

    compiled.shapeEnv = traced.shapeEnv;
    compiled.outputSymShapes = traced.outputSymShapes;
    return compiled;
  }

  function _compileSeparate(forwardFunc, traced, policy) {
    const bwdBuilder = new BackwardGraphBuilder({ rematPolicy: policy });
    const { backwardFunc, savedValues, gradInputIndices } = bwdBuilder.build(forwardFunc);

    const realReturnOp = forwardFunc.getReturnOp();
    const realOutputs = [...realReturnOp.operands];
    const numRealOutputs = realOutputs.length;

    const argIndexById = new Map(forwardFunc.args.map((a, i) => [a.id, i]));
    const outputIndexById = new Map(realOutputs.map((o, i) => [o.id, i]));

    const extraSaved = [];
    const extraIndexById = new Map();
    for (const sv of savedValues) {
      if (argIndexById.has(sv.id) || outputIndexById.has(sv.id) || extraIndexById.has(sv.id)) continue;
      extraIndexById.set(sv.id, numRealOutputs + extraSaved.length);
      extraSaved.push(sv);
    }

    if (extraSaved.length > 0) {
      realReturnOp.erase();
      new IRBuilder(forwardFunc).returnOp([...realOutputs, ...extraSaved]);
      forwardFunc.outputTypes = Object.freeze([
        ...realOutputs.map(v => v.type),
        ...extraSaved.map(v => v.type),
      ]);
    }

    const savedSources = savedValues.map(sv => {
      if (argIndexById.has(sv.id)) return { kind: 'arg', index: argIndexById.get(sv.id) };
      if (outputIndexById.has(sv.id)) return { kind: 'output', index: outputIndexById.get(sv.id) };
      return { kind: 'output', index: extraIndexById.get(sv.id) };
    });

    const fwdModule = new GraphModule('forward');
    fwdModule.addFunction(forwardFunc);
    const fwdResult = new Compiler(compilerOpts).compile(fwdModule);

    const bwdModule = new GraphModule('backward');
    bwdModule.addFunction(backwardFunc);
    const bwdResult = new Compiler(compilerOpts).compile(bwdModule);

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
    };
  }

  function _compileJoint(forwardFunc, traced, policy) {
    const jointBuilder = new JointGraphBuilder({ rematPolicy: policy });
    const { jointFunc, numForwardOutputs, numGradInputs } = jointBuilder.build(forwardFunc);

    const module = new GraphModule('joint');
    module.addFunction(jointFunc);
    const result = new Compiler(compilerOpts).compile(module);

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
    };
  }

  function _resolveOutputShape(compiled, i) {
    if (compiled.outputSymShapes && compiled.shapeEnv) {
      return compiled.shapeEnv.resolveSymbolicShape(compiled.outputSymShapes[i]);
    }
    return compiled.outputTypes[i].shape;
  }

  function _runK(result, funcName, allArgs) {
    const mod = result.module || result;
    if (mod.executionPlan) return mod.runPlanAsync(mod.executionPlan, allArgs);
    if (result.isAsync(funcName)) return result.runAsync(funcName, ...allArgs);
    result.run(funcName, ...allArgs);
    return null;
  }

  function _executeSeparateForward(compiled, inputs) {
    const kernels = compiled.fwdResult.listKernels();
    const funcName = kernels[0];
    const device = inputs.length > 0 ? inputs[0].device : 'cpu';

    const inputArrays = inputs.map(t => tensorToContiguous(t));
    const params = compiled.capturedParams;
    const paramArrays = params.map(t => tensorToContiguous(t));

    const allOutputTypes = compiled.forwardFunc.outputTypes;
    const numRealOutputs = compiled.numRealOutputs;
    const outputArrays = new Array(allOutputTypes.length);
    const outputShapes = new Array(allOutputTypes.length);
    for (let i = 0; i < allOutputTypes.length; i++) {
      const shape = i < numRealOutputs ? _resolveOutputShape(compiled, i) : allOutputTypes[i].shape;
      const dtype = allOutputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      outputArrays[i] = new Ctor(Math.max(numel, 1));
      outputShapes[i] = shape;
    }

    const allArgs = [...inputArrays, ...paramArrays, ...outputArrays];
    const build = () => {
      const results = numRealOutputs === 1
        ? wrapResult(outputArrays[0], outputShapes[0], allOutputTypes[0].dtype, device)
        : Array.from({ length: numRealOutputs }, (_, i) => wrapResult(outputArrays[i], outputShapes[i], allOutputTypes[i].dtype, device));
      return { results, inputArrays, paramArrays, outputArrays, device };
    };
    const pending = _runK(compiled.fwdResult, funcName, allArgs);
    return pending ? pending.then(build) : build();
  }

  function _executeSeparateBackward(compiled, gradOutputs, savedContext) {
    const kernels = compiled.bwdResult.listKernels();
    const funcName = kernels[0];

    const gradArrays = gradOutputs.map(t => tensorToContiguous(t));

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
    const gradInputArrays = new Array(numGradOutputs);
    const gradInputShapes = new Array(numGradOutputs);
    for (let i = 0; i < numGradOutputs; i++) {
      const shape = bwdFunc.outputTypes[i].shape;
      const dtype = bwdFunc.outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      gradInputArrays[i] = new Ctor(Math.max(numel, 1));
      gradInputShapes[i] = shape;
    }

    const allArgs = [...gradArrays, ...savedArrays, ...gradInputArrays];
    const build = () => gradInputArrays.map((arr, i) =>
      wrapResult(arr, gradInputShapes[i], bwdFunc.outputTypes[i].dtype, savedContext.device)
    );
    const pending = _runK(compiled.bwdResult, funcName, allArgs);
    return pending ? pending.then(build) : build();
  }

  function _findCachedEntry(inputs) {
    for (let i = 0; i < _cacheEntries.length; i++) {
      const entry = _cacheEntries[i];
      entry.shapeEnv.bindInputShapes(inputs);
      const { passed } = entry.shapeEnv.evaluateGuards();
      if (passed) return entry;
    }
    return null;
  }

  function compiledForward(...inputs) {
    let meta = _findCachedEntry(inputs);
    if (!meta) {
      meta = _compile(inputs);
      _cacheEntries.push(meta);
      meta.shapeEnv.bindInputShapes(inputs);
    }
    _activeMeta = meta;

    if (meta.mode === 'joint') {
      return _executeJointForward(meta, inputs);
    }

    const ctx = _executeSeparateForward(meta, inputs);
    if (ctx && ctx.then) return ctx.then((c) => { _savedValues = c; return c.results; });
    _savedValues = ctx;
    return ctx.results;
  }

  function _executeJointForward(compiled, inputs) {
    const kernels = compiled.result.listKernels();
    const funcName = kernels[0];
    const device = inputs.length > 0 ? inputs[0].device : 'cpu';

    const inputArrays = inputs.map(t => tensorToContiguous(t));
    const params = compiled.capturedParams;
    const paramArrays = params.map(t => tensorToContiguous(t));

    const jointFunc = compiled.jointFunc;
    const numOutputs = jointFunc.outputTypes.length;
    const outputArrays = new Array(numOutputs);
    const outputShapes = new Array(numOutputs);
    for (let i = 0; i < numOutputs; i++) {
      const shape = jointFunc.outputTypes[i].shape;
      const dtype = jointFunc.outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      outputArrays[i] = new Ctor(Math.max(numel, 1));
      outputShapes[i] = shape;
    }

    const gradOutputArrays = new Array(compiled.numForwardOutputs);
    for (let i = 0; i < compiled.numForwardOutputs; i++) {
      const type = compiled.outputTypes[i];
      const numel = computeNumel(type.shape);
      const Ctor = typedArrayCtor(type.dtype);
      gradOutputArrays[i] = new Ctor(Math.max(numel, 1));
    }

    _savedValues = { inputArrays, paramArrays, gradOutputArrays, outputArrays, outputShapes, device, compiled };

    const allArgs = [...inputArrays, ...paramArrays, ...gradOutputArrays, ...outputArrays];
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

  compiledForward.backward = function (...gradOutputs) {
    if (!_activeMeta || !_savedValues) {
      throw new Error('Must run forward before backward');
    }

    if (_activeMeta.mode === 'joint') {
      return _executeJointBackward(_activeMeta, gradOutputs, _savedValues);
    }

    return _executeSeparateBackward(_activeMeta, gradOutputs, _savedValues);
  };

  function _executeJointBackward(compiled, gradOutputs, savedCtx) {
    const { inputArrays, paramArrays, outputArrays, outputShapes, device } = savedCtx;

    const gradArrays = gradOutputs.map(t => tensorToContiguous(t));
    for (let i = 0; i < gradArrays.length; i++) {
      savedCtx.gradOutputArrays[i].set(gradArrays[i]);
    }

    const jointFunc = compiled.jointFunc;
    const numOutputs = jointFunc.outputTypes.length;
    const newOutputArrays = new Array(numOutputs);
    const newOutputShapes = new Array(numOutputs);
    for (let i = 0; i < numOutputs; i++) {
      const shape = jointFunc.outputTypes[i].shape;
      const dtype = jointFunc.outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      newOutputArrays[i] = new Ctor(Math.max(numel, 1));
      newOutputShapes[i] = shape;
    }

    const allArgs = [...inputArrays, ...paramArrays, ...gradArrays, ...newOutputArrays];
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

  compiledForward.original = model;

  compiledForward.backwardGraph = () => {
    if (_cacheEntries.length === 0) return null;
    const meta = _cacheEntries[0];
    if (meta.mode === 'joint') return meta.jointFunc;
    return meta.backwardFunc;
  };

  compiledForward.forwardGraph = () => {
    if (_cacheEntries.length === 0) return null;
    const meta = _cacheEntries[0];
    if (meta.mode === 'joint') return meta.jointFunc;
    return meta.forwardFunc;
  };

  if (exampleInputs) {
    const compiled = _compile(exampleInputs);
    _cacheEntries.push(compiled);
  }

  return compiledForward;
}
