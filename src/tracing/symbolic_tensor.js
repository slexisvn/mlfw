import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';
import { Storage } from '../tensor/core/storage.js';
import { DispatchKey, DispatchKeySet } from '../dispatcher/dispatch_key.js';
import { META_DEVICE } from '../tensor/types/device.js';
import { computeStrides } from '../tensor/utils/shape_utils.js';

const TRACING_KEY = DispatchKeySet.fromKey(DispatchKey.TRACING);

export class SymbolicTensor extends Tensor {
  constructor(irValue, shape, dtype, tracer) {
    const strides = computeStrides(shape);
    const storage = Storage.allocate(0, dtype, META_DEVICE);
    const impl = new TensorImpl(storage, 0, shape, strides, dtype, META_DEVICE);
    super(impl);
    this._irValue = irValue;
    this._tracer = tracer;
  }

  get irValue() {
    return this._irValue;
  }

  get tracer() {
    return this._tracer;
  }

  get dispatchKeySet() {
    return super.dispatchKeySet.union(TRACING_KEY);
  }

  get isSymbolic() {
    return true;
  }
}
