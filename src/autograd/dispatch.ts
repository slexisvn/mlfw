import { DispatchKey, AUTOGRAD_KEY_SET } from '../dispatcher/dispatch_key.js';
import { KernelFunction } from '../dispatcher/boxing.js';
import { dispatcher } from '../dispatcher/dispatcher.js';
import { GradMode } from './grad_mode.js';
import { getGradFn, isDecomposedOp, isGradientBarrier } from './registry.js';
import { GradAccumulator } from './accumulator.js';
import { AutogradMeta } from '../tensor/core/autograd_meta.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Tensor } from '../tensor/core/tensor.js';
import { isTensor } from '../tensor/core/is_tensor.js';
import { DeviceType } from '../tensor/types/device.js';
import { isEagerDeferred } from '../dispatcher/eager_mode.js';
import { setAutogradEngine } from '../tensor/core/tensor.js';
import { backward } from './engine.js';
import type { DispatchKeySet } from '../dispatcher/dispatch_key.js';
import type { OperatorHandle } from '../dispatcher/operator_handle.js';
import type { AutogradNode } from './node.js';

setAutogradEngine({ backward });

function _snapshotTensor(t: Tensor): Tensor {
  const impl = t._impl;
  if (isEagerDeferred() && impl.device && impl.device.type === DeviceType.GPU) {
    impl.storage.retain();
    const aliasImpl = new TensorImpl(
      impl.storage,
      impl.storageOffset,
      impl.sizes(),
      impl.strides(),
      impl.dtype,
      impl.device
    );
    return new Tensor(aliasImpl);
  }
  const clonedStorage = impl.storage.clone();
  const newImpl = new TensorImpl(
    clonedStorage,
    impl.storageOffset,
    impl.sizes(),
    impl.strides(),
    impl.dtype,
    impl.device
  );
  return new Tensor(newImpl);
}

function _anyRequiresGrad(args: readonly unknown[]): boolean {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (isTensor(a) && a.requiresGrad) return true;
    if (Array.isArray(a)) {
      for (let j = 0; j < a.length; j++) {
        if (isTensor(a[j]) && a[j].requiresGrad) return true;
      }
    }
  }
  return false;
}

function _extractTensors(args: readonly unknown[]): Tensor[] {
  const tensors: Tensor[] = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (isTensor(a)) {
      tensors.push(a);
    } else if (Array.isArray(a)) {
      for (let j = 0; j < a.length; j++) {
        if (isTensor(a[j])) tensors.push(a[j]);
      }
    }
  }
  return tensors;
}

function _getOrCreateAccumulator(tensor: Tensor): AutogradNode | null {
  const meta = tensor._impl.autogradMeta;
  if (!meta) return null;
  let acc = meta.getGradAccumulator() as unknown as AutogradNode | null;
  if (!acc) {
    acc = new GradAccumulator(tensor);
    meta.setGradAccumulator(acc);
  }
  return acc;
}

export function wrapWithAutograd(opName: string, handle: OperatorHandle) {
  return (keySet: unknown, ...args: unknown[]) => {
    const ks = keySet as DispatchKeySet;
    if (!GradMode.isEnabled() || !_anyRequiresGrad(args)) {
      const stripped = ks.subtract(AUTOGRAD_KEY_SET);
      return dispatcher.redispatch(handle, stripped, ...args);
    }

    const gradFn = getGradFn(opName, args);
    if (!gradFn) {
      const stripped = ks.subtract(AUTOGRAD_KEY_SET);
      return dispatcher.redispatch(handle, stripped, ...args);
    }


    const tensorArgs = _extractTensors(args);
    gradFn.setOpArgs(args);
    for (let i = 0; i < tensorArgs.length; i++) {
      gradFn.saveTensor(_snapshotTensor(tensorArgs[i]));
      gradFn.saveInputMetadata(i, [...tensorArgs[i].shape], tensorArgs[i].dtype);
    }

    for (let i = 0; i < tensorArgs.length; i++) {
      const t = tensorArgs[i];
      if (t.requiresGrad) {
          const fn = t.gradFn as AutogradNode | null;
        if (fn) {
          const outputNr = t._impl.autogradMeta ? t._impl.autogradMeta.outputNr : 0;
          gradFn.setNextEdge(i, fn, outputNr);
        } else {
          const acc = _getOrCreateAccumulator(t);
          if (acc) gradFn.setNextEdge(i, acc, 0);
        }
      }
    }

    const stripped = ks.subtract(AUTOGRAD_KEY_SET);
    const result = dispatcher.redispatch(handle, stripped, ...args);

    if (isTensor(result)) {
      result._impl.setAutogradMeta(new AutogradMeta());
      const meta = result._impl.autogradMeta;
      meta!.setGradFn(gradFn, 0);
      meta!.requiresGrad = true;
      result._impl._updateKeySet();
    } else if (Array.isArray(result) && result.some(isTensor)) {
      throw new Error(`autograd: op '${opName}' returns multiple tensors, which eager autograd cannot track — the gradient would be silently dropped. Use compileWithBackward, or detach the inputs.`);
    }

    return result;
  };
}

function _makePassthrough(handle: OperatorHandle) {
  return (keySet: unknown, ...args: unknown[]) => {
    const stripped = (keySet as DispatchKeySet).subtract(AUTOGRAD_KEY_SET);
    return dispatcher.redispatch(handle, stripped, ...args);
  };
}

export function registerAutogradKernels(): void {
  const ops = dispatcher.listOps();
  const keys = [DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU, DispatchKey.AUTOGRAD_GPU, DispatchKey.AUTOGRAD_WASM];

  for (const opKey of ops) {
    const handle = dispatcher.findOp(opKey);
    if (!handle) continue;
    const opName = handle.name;

    const fn = isGradientBarrier(opName) || isDecomposedOp(opName)
      ? _makePassthrough(handle)
      : wrapWithAutograd(opName, handle);

    const kernel = KernelFunction.fromUnboxed(fn);
    for (const key of keys) handle.entry.registerKernel(key, kernel);
  }
}
