import { Tensor } from '../tensor/core/tensor.js';
import { TensorImpl } from '../tensor/core/tensor_impl.js';

export class Parameter extends Tensor {
  constructor(data, requiresGrad = true) {
    if (data instanceof Tensor) {
      super(data._impl);
    } else {
      super(data);
    }
    if (requiresGrad) this.requiresGrad_(true);
  }

  get isParameter() {
    return true;
  }
}
