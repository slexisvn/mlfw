import { _traceCore, executeCompiled } from './compile.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../backend/target.js';
import { GraphModule } from '../compiler/ir/graph/module.js';
import { BackwardGraphBuilder } from '../compiler/ad/backward_builder.js';
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

  let _fwdCompiled = null;
  let _bwdCompiled = null;
  let _jointCompiled = null;
  let _savedValues = null;
  let _meta = null;
  let _shapeKey = null;

  function _getShapeKey(inputs) {
    let key = '';
    for (let i = 0; i < inputs.length; i++) {
      key += inputs[i].shape.join(',') + ':' + inputs[i].dtype + '|';
    }
    return key;
  }

  function _compile(inputs) {
    const traced = _traceCore(
      (...args) => model.forward(...args),
      inputs,
      { name: model.constructor.name || 'compiled' }
    );

    const graph = traced.graph;
    const func = graph.functions().next().value;

    if (mode === 'joint') {
      return _compileJoint(func, traced, rematPolicy);
    }
    return _compileSeparate(func, traced, rematPolicy);
  }

  function _compileSeparate(forwardFunc, traced, policy) {
    const bwdBuilder = new BackwardGraphBuilder({ rematPolicy: policy });
    const { backwardFunc, savedValues, gradInputIndices } = bwdBuilder.build(forwardFunc);

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

  function _executeSeparateForward(compiled, inputs) {
    const kernels = compiled.fwdResult.listKernels();
    const funcName = kernels[0];
    const device = inputs.length > 0 ? inputs[0].device : 'cpu';

    const inputArrays = inputs.map(t => tensorToContiguous(t));
    const params = compiled.capturedParams;
    const paramArrays = params.map(t => tensorToContiguous(t));

    const outputTypes = compiled.outputTypes;
    const outputArrays = new Array(outputTypes.length);
    const outputShapes = new Array(outputTypes.length);
    for (let i = 0; i < outputTypes.length; i++) {
      const shape = outputTypes[i].shape;
      const dtype = outputTypes[i].dtype;
      const numel = computeNumel(shape);
      const Ctor = typedArrayCtor(dtype);
      outputArrays[i] = new Ctor(Math.max(numel, 1));
      outputShapes[i] = shape;
    }

    const allArgs = [...inputArrays, ...paramArrays, ...outputArrays];
    compiled.fwdResult.run(funcName, ...allArgs);

    const results = outputTypes.length === 1
      ? wrapResult(outputArrays[0], outputShapes[0], outputTypes[0].dtype, device)
      : outputArrays.map((arr, i) => wrapResult(arr, outputShapes[i], outputTypes[i].dtype, device));

    return { results, savedBuffers: [...inputArrays, ...paramArrays, ...outputArrays], device };
  }

  function _executeSeparateBackward(compiled, gradOutputs, savedContext) {
    const kernels = compiled.bwdResult.listKernels();
    const funcName = kernels[0];

    const gradArrays = gradOutputs.map(t => tensorToContiguous(t));

    const savedValues = compiled.savedValues;
    const savedArrays = new Array(savedValues.length);
    for (let i = 0; i < savedValues.length; i++) {
      const idx = savedValues[i].id;
      savedArrays[i] = savedContext.savedBuffers[i] || new Float32Array(computeNumel(savedValues[i].type.shape));
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
    compiled.bwdResult.run(funcName, ...allArgs);

    return gradInputArrays.map((arr, i) =>
      wrapResult(arr, gradInputShapes[i], bwdFunc.outputTypes[i].dtype, savedContext.device)
    );
  }

  function compiledForward(...inputs) {
    const key = _getShapeKey(inputs);
    if (!_meta || _shapeKey !== key) {
      _meta = _compile(inputs);
      _shapeKey = key;
    }

    if (_meta.mode === 'joint') {
      return _executeJointForward(_meta, inputs);
    }

    const ctx = _executeSeparateForward(_meta, inputs);
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
    compiled.result.run(funcName, ...allArgs);

    const fwdOutputs = [];
    for (let i = 0; i < compiled.numForwardOutputs; i++) {
      fwdOutputs.push(wrapResult(outputArrays[i], outputShapes[i], jointFunc.outputTypes[i].dtype, device));
    }

    return fwdOutputs.length === 1 ? fwdOutputs[0] : fwdOutputs;
  }

  compiledForward.backward = function (...gradOutputs) {
    if (!_meta || !_savedValues) {
      throw new Error('Must run forward before backward');
    }

    if (_meta.mode === 'joint') {
      return _executeJointBackward(_meta, gradOutputs, _savedValues);
    }

    return _executeSeparateBackward(_meta, gradOutputs, _savedValues);
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
    compiled.result.run(compiled.result.listKernels()[0], ...allArgs);

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
  }

  compiledForward.original = model;

  compiledForward.backwardGraph = () => {
    if (!_meta) return null;
    if (_meta.mode === 'joint') return _meta.jointFunc;
    return _meta.backwardFunc;
  };

  compiledForward.forwardGraph = () => {
    if (!_meta) return null;
    if (_meta.mode === 'joint') return _meta.jointFunc;
    return _meta.forwardFunc;
  };

  if (exampleInputs) {
    _meta = _compile(exampleInputs);
    _shapeKey = _getShapeKey(exampleInputs);
  }

  return compiledForward;
}
