import { AutogradNode } from './node.js';
import { add as _add } from '../tensor/ops/ops.js';

export class GradAccumulator extends AutogradNode {
  constructor(variable) {
    super(0);
    this._variable = typeof WeakRef !== 'undefined' ? new WeakRef(variable) : { deref: () => variable };
  }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const tensor = this._variable.deref();
    if (!tensor) return [];

    const meta = tensor._impl.autogradMeta;
    if (!meta) return [];

    if (meta.grad === null) {
      meta.grad = g;
    } else {
      meta.grad = _add(meta.grad, g);
    }

    return [];
  }

  name() {
    return 'GradAccumulator';
  }
}
