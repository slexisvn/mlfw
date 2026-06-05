import { Tracer } from './tracer.js';
import { registerTracingDispatch } from './dispatch.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { withIncludedKeys } from '../dispatcher/guard.js';
import { Compiler } from '../compiler/pipeline/compiler.js';
import { CPUTarget } from '../backend/target.js';
import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { computeStrides, computeNumel } from '../tensor/utils/shape_utils.js';

let _tracingRegistered = false;

function _ensureTracing() {
  if (!_tracingRegistered) {
    registerTracingDispatch();
    _tracingRegistered = true;
  }
}

export function trace(fn, exampleInputs, opts) {
  _ensureTracing();

  const name = opts?.name || fn.name || 'traced';
  const tracer = new Tracer(name);

  for (const input of exampleInputs) {
    tracer.createInput(input.shape, input.dtype);
  }

  const symbolicInputs = tracer._initGraph();

  tracer.activate();
  try {
    const TRACING_KEYS = DispatchKeySet.fromKey(DispatchKey.TRACING);
    const result = withIncludedKeys(TRACING_KEYS, () => fn(...symbolicInputs));

    if (Array.isArray(result)) {
      tracer.markOutputs(result);
    } else {
      tracer.markOutput(result);
    }
  } finally {
    tracer.deactivate();
  }

  return tracer.getGraphModule();
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

  function _compile(inputs) {
    const graph = trace(
      (...args) => model.forward(...args),
      inputs,
      { name: model.constructor.name || 'compiled' }
    );
    return { result: new Compiler(compilerOpts).compile(graph), graph };
  }

  function compiledForward(...inputs) {
    const key = _getShapeKey(inputs);
    if (!_compiled || _shapeKey !== key) {
      _compiled = _compile(inputs);
      _shapeKey = key;
    }
    return _compiled;
  }

  if (exampleInputs) {
    _compiled = _compile(exampleInputs);
    _shapeKey = _getShapeKey(exampleInputs);
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

  return compiledForward;
}
