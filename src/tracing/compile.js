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

let _tracingRegistered = false;

function _ensureTracing() {
  if (!_tracingRegistered) {
    registerTracingDispatch();
    _tracingRegistered = true;
  }
}

function _normalizeDynamicShapes(dynamicShapes, exampleInputs) {
  if (!dynamicShapes) return new Array(exampleInputs.length).fill(null);

  const result = new Array(exampleInputs.length);
  for (let i = 0; i < exampleInputs.length; i++) {
    const spec = dynamicShapes[i];
    if (spec === true) {
      const all = new Set();
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

export function _traceCore(fn, exampleInputs, opts) {
  _ensureTracing();

  const name = opts?.name || fn.name || 'traced';
  const tracer = new Tracer(name);
  const dynamicShapes = _normalizeDynamicShapes(opts?.dynamicShapes, exampleInputs);

  for (let i = 0; i < exampleInputs.length; i++) {
    tracer.createInput(exampleInputs[i].shape, exampleInputs[i].dtype, dynamicShapes[i]);
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
      shapeEnv: tracer.shapeEnv,
      outputSymShapes: tracer.outputSymShapes,
    };
  }

  tracer.activate();
  const TRACING_KEYS = DispatchKeySet.fromKey(DispatchKey.TRACING);
  const result = withIncludedKeys(TRACING_KEYS, () => fn(...symbolicInputs));

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

function _prepareExecution(compiled, inputs, shapeEnv) {
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
  const outputSymShapes = compiled.outputSymShapes;
  const outputArrays = new Array(outputTypes.length);
  const outputShapes = new Array(outputTypes.length);

  for (let i = 0; i < outputTypes.length; i++) {
    const shape = outputSymShapes && shapeEnv
      ? shapeEnv.resolveSymbolicShape(outputSymShapes[i])
      : outputTypes[i].shape;
    const dtype = outputTypes[i].dtype;
    const numel = computeNumel(shape);
    const Ctor = typedArrayCtor(dtype);
    outputArrays[i] = new Ctor(Math.max(numel, 1));
    outputShapes[i] = shape;
  }

  const allArgs = new Array(inputArrays.length + paramArrays.length + outputArrays.length);
  let idx = 0;
  for (let i = 0; i < inputArrays.length; i++) allArgs[idx++] = new RuntimeTensor(inputArrays[i], inputs[i].shape, inputs[i].dtype);
  for (let i = 0; i < paramArrays.length; i++) {
    const rt = new RuntimeTensor(paramArrays[i], params[i].shape, params[i].dtype);
    const impl = params[i]._impl;
    if (impl) rt.resident = { key: impl.storage.rawData, version: impl.version };
    allArgs[idx++] = rt;
  }
  for (let i = 0; i < outputArrays.length; i++) allArgs[idx++] = new RuntimeTensor(outputArrays[i], outputShapes[i], outputTypes[i].dtype);

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

export function executeCompiled(compiled, inputs, shapeEnv) {
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

export function compile(model, exampleInputs, opts) {
  if (opts?.backward) {
    return _compileWithBackward(model, exampleInputs, opts);
  }

  const target = opts?.target ?? CPUTarget();
  const compilerOpts = { target, verify: false, ...opts };
  const dynamicShapes = opts?.dynamic_shapes || null;
  const shapeBuckets = opts?.shapeBuckets || null;
  const foldWeights = opts?.foldWeights ?? opts?.quantization?.foldWeights ?? false;

  const _cacheEntries = [];

  function _attachRepro(error, inputs, phase) {
    if (!error || typeof error !== 'object' || error.repro) return error;
    try {
      error.repro = {
        name: model.constructor?.name || 'compiled',
        phase,
        target: target?.name,
        inputs: (inputs || []).map(t => ({ shape: t.shape, dtype: t.dtype })),
        config: {
          fusion: compilerOpts.fusion,
          scheduling: compilerOpts.scheduling,
          optimization: compilerOpts.optimization,
          quantization: compilerOpts.quantization,
          dynamicShapes: !!dynamicShapes,
        },
      };
    } catch (_) { return error; }
    return error;
  }

  function _finalize(traced) {
    const prepared = foldWeights ? foldWeightParams(traced, tensorToContiguous) : traced;
    const result = new Compiler(compilerOpts).compile(prepared.graph);
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

  function _compileWith(inputs, dynShapes) {
    try {
      const traced = _traceCore(
        (...args) => model.forward(...args),
        inputs,
        { name: model.constructor.name || 'compiled', dynamicShapes: dynShapes }
      );
      if (traced && typeof traced.then === 'function') {
        return traced.then(_finalize, e => { throw _attachRepro(e, inputs, 'compile'); });
      }
      return _finalize(traced);
    } catch (e) {
      throw _attachRepro(e, inputs, 'compile');
    }
  }

  function _compile(inputs) {
    return _compileWith(inputs, dynamicShapes);
  }

  function _bucketInputs(shapes) {
    return shapes.map((shape, i) => ({ shape, dtype: exampleInputs[i].dtype }));
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

  function _execute(entry, inputs) {
    try {
      const out = executeCompiled(entry, inputs, entry.shapeEnv);
      if (out && typeof out.then === 'function') {
        return out.then(undefined, e => { throw _attachRepro(e, inputs, 'run'); });
      }
      return out;
    } catch (e) {
      throw _attachRepro(e, inputs, 'run');
    }
  }

  function compiledForward(...inputs) {
    let entry = _findCachedEntry(inputs);
    if (!entry) {
      const result = _compile(inputs);
      if (result && typeof result.then === 'function') {
        return result.then(c => {
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

  let _ready = null;
  if (exampleInputs) {
    const pending = [];
    if (shapeBuckets) {
      for (const bucketShapes of shapeBuckets) {
        pending.push(_compileWith(_bucketInputs(bucketShapes), null));
      }
    }
    pending.push(_compile(exampleInputs));

    if (pending.some(r => r && typeof r.then === 'function')) {
      _ready = Promise.all(pending).then(entries => { for (const c of entries) _cacheEntries.push(c); });
    } else {
      for (const c of pending) _cacheEntries.push(c);
    }
  }

  compiledForward.original = model;

  compiledForward.graph = (inputs) => {
    const args = inputs || exampleInputs;
    return trace(
      (...a) => model.forward(...a),
      args,
      { name: model.constructor.name || 'compiled', dynamicShapes }
    );
  };

  compiledForward.source = () => {
    if (_cacheEntries.length === 0) return null;
    const compiled = _cacheEntries[0];
    const kernels = compiled.result.listKernels();
    return kernels.length > 0 ? compiled.result.getSource(kernels[0]) : null;
  };

  compiledForward.kernels = () => {
    if (_cacheEntries.length === 0) return [];
    return _cacheEntries[0].result.listKernels();
  };

  compiledForward.snippet = () => {
    if (_cacheEntries.length === 0) return null;
    const compiled = _cacheEntries[0];
    const kernels = compiled.result.listKernels();
    return kernels.length > 0 ? compiled.result.getSnippet(kernels[0]) : null;
  };

  compiledForward.result = () => _cacheEntries.length > 0 ? _cacheEntries[0].result : null;
  compiledForward._ready = _ready;

  return compiledForward;
}
