import { DispatchKey } from '../dispatcher/dispatch_key.js';
import { KernelFunction } from '../dispatcher/boxing.js';
import { dispatcher } from '../dispatcher/dispatcher.js';
import { getActiveTracer } from './tracer.js';
import { SymbolicTensor } from './symbolic_tensor.js';
import { scalarArgNames } from '../tensor/ops/metadata.js';

const _TRACE_BY_DECOMPOSITION = new Set([
  'scatter', 'repeat', 'tile', 'split', 'chunk', 'roll',
  'flip', 'cumsum', 'sort', 'argsort', 'topk',
]);

function _tracingKernel(opName) {
  return (keySet, ...args) => {
    const tracer = getActiveTracer();
    if (!tracer) {
      throw new Error(`TRACING dispatch key active but no tracer is set for op '${opName}'`);
    }

    if (_TRACE_BY_DECOMPOSITION.has(opName)) {
      return dispatcher.redispatch(dispatcher.findOp(opName), keySet, ...args);
    }

    const tensorArgs = [];
    const scalarArgs = {};
    const spec = scalarArgNames(opName);
    let scalarIdx = 0;

    const isTensor = (a) => (a instanceof SymbolicTensor) || (a && a._impl);
    const pushTensor = (arg) => {
      if (arg instanceof SymbolicTensor) {
        tensorArgs.push(arg);
      } else if (arg && arg._impl && arg.isSymbolic) {
        tensorArgs.push(arg);
      } else if (arg && arg._impl) {
        tensorArgs.push(tracer.captureConstant(arg));
      }
    };

    for (const arg of args) {
      if (Array.isArray(arg) && arg.length > 0 && isTensor(arg[0])) {
        for (const el of arg) pushTensor(el);
      } else if (isTensor(arg)) {
        pushTensor(arg);
      } else if (spec) {
        if (arg !== undefined && arg !== null && scalarIdx < spec.length) {
          scalarArgs[spec[scalarIdx]] = arg;
        }
        scalarIdx++;
      }
    }

    return tracer.recordOp(opName, tensorArgs, scalarArgs);
  };
}

export function registerTracingDispatch() {
  const ops = dispatcher.listOps();
  for (const opKey of ops) {
    const handle = dispatcher.findOp(opKey);
    if (!handle) continue;
    const opName = handle.name;

    const kernel = KernelFunction.fromUnboxed(_tracingKernel(opName));
    handle.entry.registerKernel(DispatchKey.TRACING, kernel);
  }
}
