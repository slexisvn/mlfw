import { Tracer } from './tracer.js';
import { registerTracingDispatch } from './dispatch.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { withIncludedKeys } from '../dispatcher/guard.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../backend/target.js';
import { tensorToContiguous, wrapResult } from '../dispatcher/jit_dispatch.js';
import { typedArrayCtor } from '../tensor/types/dtype.js';
import { computeNumel } from '../tensor/utils/shape_utils.js';

let _tracingRegistered = false;

function _ensureTracing() {
  if (!_tracingRegistered) {
    registerTracingDispatch();
    _tracingRegistered = true;
  }
}

export function _traceCore(fn, exampleInputs, opts) {
  _ensureTracing();

  const name = opts?.name || fn.name || 'traced';
  const tracer = new Tracer(name);

  for (const input of exampleInputs) {
    tracer.createInput(input.shape, input.dtype);
  }

  const numUserInputs = exampleInputs.length;
  const symbolicInputs = tracer._initGraph();

  function _finalize(result) {
    if (Array.isArray(result)) {
      tracer.markOutputs(result);
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
    };
  }

  tracer.activate();
  const TRACING_KEYS = DispatchKeySet.fromKey(DispatchKey.TRACING);
  const result = withIncludedKeys(TRACING_KEYS, () => fn(...symbolicInputs));

  // Support async forward functions (e.g. Tensor Lang custom models)
  if (result && typeof result.then === 'function') {
    return result.then(
      resolved => _finalize(resolved),
      error => { tracer.deactivate(); throw error; },
    );
  }

  try {
    return _finalize(result);
  } catch (error) {
    tracer.deactivate();
    throw error;
  }
}

export function trace(fn, exampleInputs, opts) {
  const result = _traceCore(fn, exampleInputs, opts);
  if (result && typeof result.then === 'function') {
    return result.then(r => r.graph);
  }
  return result.graph;
}

function _prepareExecution(compiled, inputs) {
  const kernels = compiled.result.listKernels();
  if (kernels.length === 0) throw new Error('No kernels compiled');
  const funcName = kernels[0];

  const device = inputs.length > 0 ? inputs[0].device : 'cpu';

  const inputArrays = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    inputArrays[i] = tensorToContiguous(inputs[i]);
  }

  const params = compiled.capturedParams;
  const paramArrays = new Array(params.length);
  for (let i = 0; i < params.length; i++) {
    paramArrays[i] = tensorToContiguous(params[i]);
  }

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

  const allArgs = new Array(inputArrays.length + paramArrays.length + outputArrays.length);
  let idx = 0;
  for (let i = 0; i < inputArrays.length; i++) allArgs[idx++] = inputArrays[i];
  for (let i = 0; i < paramArrays.length; i++) allArgs[idx++] = paramArrays[i];
  for (let i = 0; i < outputArrays.length; i++) allArgs[idx++] = outputArrays[i];

  return { funcName, device, outputTypes, outputArrays, outputShapes, allArgs };
}

function _wrapOutputs(device, outputTypes, outputArrays, outputShapes) {
  if (outputTypes.length === 1) {
    return wrapResult(outputArrays[0], outputShapes[0], outputTypes[0].dtype, device);
  }
  const results = new Array(outputTypes.length);
  for (let i = 0; i < outputTypes.length; i++) {
    results[i] = wrapResult(outputArrays[i], outputShapes[i], outputTypes[i].dtype, device);
  }
  return results;
}

export function executeCompiled(compiled, inputs) {
  const { funcName, device, outputTypes, outputArrays, outputShapes, allArgs } = _prepareExecution(compiled, inputs);

  if (compiled.result.isAsync(funcName)) {
    return compiled.result.runAsync(funcName, ...allArgs)
      .then(() => _wrapOutputs(device, outputTypes, outputArrays, outputShapes));
  }

  compiled.result.run(funcName, ...allArgs);
  return _wrapOutputs(device, outputTypes, outputArrays, outputShapes);
}

export function compile(model, exampleInputs, opts) {
  const target = opts?.target ?? CPUTarget();
  const compilerOpts = { target, verify: false, ...opts };

  let _compiled = null;
  let _shapeKey = null;

  function _getShapeKey(inputs) {
    let key = '';
    for (let i = 0; i < inputs.length; i++) {
      const inp = inputs[i];
      key += inp.shape.join(',') + ':' + inp.dtype + '|';
    }
    return key;
  }

  function _finalize(traced) {
    const result = new Compiler(compilerOpts).compile(traced.graph);
    return {
      result,
      graph: traced.graph,
      capturedParams: traced.capturedParams,
      numUserInputs: traced.numUserInputs,
      outputTypes: traced.outputTypes,
    };
  }

  function _compile(inputs) {
    const traced = _traceCore(
      (...args) => model.forward(...args),
      inputs,
      { name: model.constructor.name || 'compiled' }
    );
    if (traced && typeof traced.then === 'function') {
      return traced.then(_finalize);
    }
    return _finalize(traced);
  }

  function compiledForward(...inputs) {
    const key = _getShapeKey(inputs);
    if (!_compiled || _shapeKey !== key) {
      const result = _compile(inputs);
      if (result && typeof result.then === 'function') {
        return result.then(c => { _compiled = c; _shapeKey = key; return executeCompiled(_compiled, inputs); });
      }
      _compiled = result;
      _shapeKey = key;
    }
    return executeCompiled(_compiled, inputs);
  }

  let _ready = null;
  if (exampleInputs) {
    const result = _compile(exampleInputs);
    if (result && typeof result.then === 'function') {
      _ready = result.then(c => { _compiled = c; _shapeKey = _getShapeKey(exampleInputs); });
    } else {
      _compiled = result;
      _shapeKey = _getShapeKey(exampleInputs);
    }
  }

  compiledForward.original = model;

  compiledForward.graph = (inputs) => {
    const args = inputs || exampleInputs;
    return trace(
      (...a) => model.forward(...a),
      args,
      { name: model.constructor.name || 'compiled' }
    );
  };

  compiledForward.source = () => {
    if (!_compiled) return null;
    const kernels = _compiled.result.listKernels();
    return kernels.length > 0 ? _compiled.result.getSource(kernels[0]) : null;
  };

  compiledForward.kernels = () => {
    if (!_compiled) return [];
    return _compiled.result.listKernels();
  };

  compiledForward.snippet = () => {
    if (!_compiled) return null;
    const kernels = _compiled.result.listKernels();
    return kernels.length > 0 ? _compiled.result.getSnippet(kernels[0]) : null;
  };

  compiledForward.result = () => _compiled ? _compiled.result : null;
  compiledForward._ready = _ready;

  return compiledForward;
}
