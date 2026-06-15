import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { ones, full } from '../../tensor/factory/creation_ops.js';

export class ExpBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    return [ops.mul(gradOutputs[0], ops.exp(input.detach()))];
  }
}

export class LogBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    return [ops.div(gradOutputs[0], input.detach())];
  }
}

export class SqrtBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const x = input.detach();
    const sqrtX = ops.sqrt(x);
    const two = full(x.shape, 2, { dtype: x.dtype, device: x.device });
    return [ops.div(gradOutputs[0], ops.mul(two, sqrtX))];
  }
}

export class TanhBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const x = input.detach();
    const th = ops.tanh(x);
    const one = ones(x.shape, { dtype: x.dtype, device: x.device });
    return [ops.mul(gradOutputs[0], ops.sub(one, ops.mul(th, th)))];
  }
}

export class SigmoidBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const x = input.detach();
    const sig = ops.sigmoid(x);
    const one = ones(x.shape, { dtype: x.dtype, device: x.device });
    return [ops.mul(gradOutputs[0], ops.mul(sig, ops.sub(one, sig)))];
  }
}

export class SoftmaxBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const args = this.opArgs();
    const dim = (args && args.length > 1 && args[1] != null) ? args[1] : -1;
    const g = gradOutputs[0];
    const y = ops.softmax(input.detach(), dim);
    const inner = ops.sum(ops.mul(g, y), dim, true);
    return [ops.mul(y, ops.sub(g, inner))];
  }
}

export class LogSoftmaxBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const args = this.opArgs();
    const dim = (args && args.length > 1 && args[1] != null) ? args[1] : -1;
    const g = gradOutputs[0];
    const sm = ops.softmax(input.detach(), dim);
    const gsum = ops.sum(g, dim, true);
    return [ops.sub(g, ops.mul(sm, gsum))];
  }
}

export class ReluBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const x = input.detach();
    return [ops.mul(gradOutputs[0], ops.sign(ops.relu(x)))];
  }
}

export class GeluBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [input] = this.savedTensors();
    const x = input.detach();
    const s = x.shape;
    const d = x.dtype;
    const dev = x.device;

    const c = full(s, 1.702, { dtype: d, device: dev });
    const one = ones(s, { dtype: d, device: dev });
    const cx = ops.mul(c, x);
    const sig = ops.sigmoid(cx);
    const grad = ops.mul(sig, ops.add(one, ops.mul(cx, ops.sub(one, sig))));
    return [ops.mul(g, grad)];
  }
}

export class SiluBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [input] = this.savedTensors();
    const x = input.detach();
    const sig = ops.sigmoid(x);
    const one = ones(x.shape, { dtype: x.dtype, device: x.device });
    const grad = ops.mul(sig, ops.add(one, ops.mul(x, ops.sub(one, sig))));
    return [ops.mul(g, grad)];
  }
}
