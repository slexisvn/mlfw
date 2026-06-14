import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { ones } from '../../tensor/factory/creation_ops.js';

export class AddBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    return [g, g];
  }
}

export class SubBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    return [g, ops.neg(g)];
  }
}

export class MulBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [self, other] = this.savedTensors();
    return [ops.mul(g, other.detach()), ops.mul(g, self.detach())];
  }
}

export class DivBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [self, other] = this.savedTensors();
    const gradSelf = ops.div(g, other.detach());
    const gradOther = ops.neg(ops.div(ops.mul(g, self.detach()), ops.mul(other.detach(), other.detach())));
    return [gradSelf, gradOther];
  }
}

export class NegBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    return [ops.neg(g)];
  }
}

export class PowBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [self, exponent] = this.savedTensors();
    const one = ones(exponent.shape, { dtype: exponent.dtype, device: exponent.device });
    const gradSelf = ops.mul(g, ops.mul(exponent.detach(), ops.pow(self.detach(), ops.sub(exponent.detach(), one))));
    const gradExp = ops.mul(g, ops.mul(ops.pow(self.detach(), exponent.detach()), ops.log(self.detach())));
    return [gradSelf, gradExp];
  }
}
