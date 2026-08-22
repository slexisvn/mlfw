import { DispatchKey } from '../dispatcher/dispatch_key.js';
import { KernelFunction } from '../dispatcher/boxing.js';
import { dispatcher } from '../dispatcher/dispatcher.js';
import { getActiveTracer } from './tracer.js';
import { SymbolicTensor } from './symbolic_tensor.js';
import { scalarArgNames } from '../tensor/ops/metadata.js';
import { isTensor as isTensorValue } from '../tensor/core/is_tensor.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { AttrMap, TensorOutput } from './types.js';
import type { DispatchKeySet as DispatchKeySetType } from '../dispatcher/dispatch_key.js';

const _TRACE_BY_DECOMPOSITION = new Set([
  'scatter', 'repeat', 'tile', 'split', 'chunk', 'roll',
  'flip', 'cumsum', 'sort', 'argsort', 'topk',
]);

type TensorCandidate = Tensor & { isSymbolic?: boolean };

function _tracingKernel(opName: string): (keySet: unknown, ...args: unknown[]) => TensorOutput | TensorOutput[] {
  return (keySet: unknown, ...args: unknown[]) => {
    const tracer = getActiveTracer();
    if (!tracer) {
      throw new Error(`TRACING dispatch key active but no tracer is set for op '${opName}'`);
    }

    if (_TRACE_BY_DECOMPOSITION.has(opName)) {
      const handle = dispatcher.findOp(opName)!;
      return dispatcher.redispatch(handle, keySet as DispatchKeySetType, ...args) as TensorOutput | TensorOutput[];
    }

    const tensorArgs: TensorOutput[] = [];
    const scalarArgs: AttrMap = {};
    const spec = scalarArgNames(opName);
    let scalarIdx = 0;

    const isTensorArg = (a: unknown): a is TensorCandidate | SymbolicTensor => (a instanceof SymbolicTensor) || isTensorValue(a);
    const pushTensor = (arg: TensorCandidate | SymbolicTensor): void => {
      if (arg instanceof SymbolicTensor) {
        tensorArgs.push(arg);
      } else if (arg.isSymbolic) {
        tensorArgs.push(arg);
      } else {
        tensorArgs.push(tracer.captureConstant(arg));
      }
    };

    for (const arg of args) {
      if (Array.isArray(arg) && arg.length > 0 && isTensorArg(arg[0])) {
        for (const el of arg) pushTensor(el);
      } else if (isTensorArg(arg)) {
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
