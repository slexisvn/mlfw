import { DispatchKey } from '../dispatcher/dispatch_key.js';
import { KernelFunction } from '../dispatcher/boxing.js';
import { dispatcher } from '../dispatcher/dispatcher.js';
import { getActiveTracer } from './tracer.js';
import { SymbolicTensor } from './symbolic_tensor.js';

const _SCALAR_ARG_SPEC = {
  sum: ['dim', 'keepdim'],
  mean: ['dim', 'keepdim'],
  max: ['dim', 'keepdim'],
  min: ['dim', 'keepdim'],
  prod: ['dim', 'keepdim'],
  argmax: ['dim', 'keepdim'],
  argmin: ['dim', 'keepdim'],
  transpose: ['dim0', 'dim1'],
  softmax: ['dim'],
  log_softmax: ['dim'],
  layer_norm: ['axis', 'eps'],
  batch_norm: ['axis', 'eps'],
  conv2d: ['strides', 'padding', 'dilation', 'groups'],
  pool2d: ['pool_type', 'kernel_size', 'strides', 'padding'],
};

function _tracingKernel(opName) {
  return (keySet, ...args) => {
    const tracer = getActiveTracer();
    if (!tracer) {
      throw new Error(`TRACING dispatch key active but no tracer is set for op '${opName}'`);
    }

    const tensorArgs = [];
    const scalarArgs = {};
    const spec = _SCALAR_ARG_SPEC[opName];
    let scalarIdx = 0;

    for (const arg of args) {
      if (arg instanceof SymbolicTensor) {
        tensorArgs.push(arg);
      } else if (arg && arg._impl && arg.isSymbolic) {
        tensorArgs.push(arg);
      } else if (arg && arg._impl) {
        tensorArgs.push(tracer.captureConstant(arg));
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
