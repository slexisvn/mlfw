import { DispatchKey, AUTOGRAD_KEY_SET } from '../dispatcher/dispatch_key.js';
import { KernelFunction } from '../dispatcher/boxing.js';
import { dispatcher } from '../dispatcher/dispatcher.js';
import { GradMode } from './grad_mode.js';
import { getGradFn, hasGradFn } from './registry.js';
import { GradAccumulator } from './accumulator.js';
import { AutogradMeta } from '../tensor/core/autograd_meta.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Tensor } from '../tensor/core/tensor.js';
import { DeviceType } from '../tensor/types/device.js';
import { isEagerDeferred } from '../dispatcher/eager_mode.js';
import { setAutogradEngine } from '../tensor/core/tensor.js';
import { backward } from './engine.js';

setAutogradEngine({ backward });

function _snapshotTensor(t) {
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

function _anyRequiresGrad(args) {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a && a._impl && a.requiresGrad) return true;
    if (Array.isArray(a)) {
      for (let j = 0; j < a.length; j++) {
        if (a[j] && a[j]._impl && a[j].requiresGrad) return true;
      }
    }
  }
  return false;
}

function _extractTensors(args) {
  const tensors = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a && a._impl) {
      tensors.push(a);
    } else if (Array.isArray(a)) {
      for (let j = 0; j < a.length; j++) {
        if (a[j] && a[j]._impl) tensors.push(a[j]);
      }
    }
  }
  return tensors;
}

function _getOrCreateAccumulator(tensor) {
  const meta = tensor._impl.autogradMeta;
  if (!meta) return null;
  let acc = meta.getGradAccumulator();
  if (!acc) {
    acc = new GradAccumulator(tensor);
    meta.setGradAccumulator(acc);
  }
  return acc;
}

export function wrapWithAutograd(opName, handle) {
  const origDispatch = dispatcher.dispatch.bind(dispatcher);

  return (keySet, ...args) => {
    if (!GradMode.isEnabled() || !_anyRequiresGrad(args)) {
      const stripped = keySet.subtract(AUTOGRAD_KEY_SET);
      return dispatcher.redispatch(handle, stripped, ...args);
    }

    const gradFn = getGradFn(opName, args);
    if (!gradFn) {
      const stripped = keySet.subtract(AUTOGRAD_KEY_SET);
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
        const fn = t.gradFn;
        if (fn) {
          const outputNr = t._impl.autogradMeta ? t._impl.autogradMeta.outputNr : 0;
          gradFn.setNextEdge(i, fn, outputNr);
        } else {
          const acc = _getOrCreateAccumulator(t);
          if (acc) gradFn.setNextEdge(i, acc, 0);
        }
      }
    }

    const stripped = keySet.subtract(AUTOGRAD_KEY_SET);
    const result = dispatcher.redispatch(handle, stripped, ...args);

    if (result && result._impl) {
      result._impl.setAutogradMeta(new AutogradMeta());
      const meta = result._impl.autogradMeta;
      meta.setGradFn(gradFn, 0);
      meta.requiresGrad = true;
      result._impl._updateKeySet();
    }

    return result;
  };
}

function _makePassthrough(handle) {
  return (keySet, ...args) => {
    const stripped = keySet.subtract(AUTOGRAD_KEY_SET);
    return dispatcher.redispatch(handle, stripped, ...args);
  };
}

export function registerAutogradKernels() {
  const ops = dispatcher.listOps();
  const keys = [DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU, DispatchKey.AUTOGRAD_GPU, DispatchKey.AUTOGRAD_WASM];

  for (const opKey of ops) {
    const handle = dispatcher.findOp(opKey);
    if (!handle) continue;
    const opName = handle.name;

    const fn = hasGradFn(opName)
      ? wrapWithAutograd(opName, handle)
      : _makePassthrough(handle);

    const kernel = KernelFunction.fromUnboxed(fn);
    for (const key of keys) handle.entry.registerKernel(key, kernel);
  }
}
