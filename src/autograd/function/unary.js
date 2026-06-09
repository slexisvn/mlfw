import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { ones, full, zeros } from '../../tensor/factory/creation_ops.js';

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

export class ReluBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs) {
    const [input] = this.savedTensors();
    const x = input.detach();
    const mask = zeros(x.shape, { dtype: x.dtype, device: x.device });
    const maskData = mask._impl.storage.data;
    const xData = x._impl.storage.data;
    for (let i = 0; i < maskData.length; i++) {
      maskData[i] = xData[i] > 0 ? 1 : 0;
    }
    return [ops.mul(gradOutputs[0], mask)];
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

    const coeff = full(s, 0.7978845608028654, { dtype: d, device: dev });
    const ic = full(s, 0.044715, { dtype: d, device: dev });
    const half = full(s, 0.5, { dtype: d, device: dev });
    const one = ones(s, { dtype: d, device: dev });
    const three = full(s, 3, { dtype: d, device: dev });

    const x3 = ops.mul(ops.mul(x, x), x);
    const inner = ops.mul(coeff, ops.add(x, ops.mul(ic, x3)));
    const th = ops.tanh(inner);

    const sechSq = ops.sub(one, ops.mul(th, th));
    const dInner = ops.mul(coeff, ops.add(one, ops.mul(ops.mul(ic, three), ops.mul(x, x))));
    const grad = ops.mul(half, ops.add(ops.add(one, th), ops.mul(x, ops.mul(sechSq, dInner))));
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
