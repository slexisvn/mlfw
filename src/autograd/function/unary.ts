import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { ones, full } from '../../tensor/factory/creation_ops.js';
import { DIGAMMA_SHIFT, DIGAMMA_SERIES } from '../../util/special_math.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { GradInputList, GradOutputList } from '../types.js';

const _TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);

function _erfDeriv(x: Tensor): Tensor {
  const coeff = full(x.shape, _TWO_OVER_SQRT_PI, { dtype: x.dtype, device: x.device });
  return ops.mul(coeff, ops.exp(ops.neg(ops.mul(x, x))));
}

function _digammaTensor(x: Tensor): Tensor {
  const z = ops.add(x, DIGAMMA_SHIFT);
  const invZ = ops.pow(z, -1);
  let acc = ops.sub(ops.log(z), ops.mul(invZ, 0.5));
  const inv2 = ops.mul(invZ, invZ);
  let p = inv2;
  for (const c of DIGAMMA_SERIES) {
    acc = ops.add(acc, ops.mul(p, c));
    p = ops.mul(p, inv2);
  }
  for (let j = 0; j < DIGAMMA_SHIFT; j++) {
    acc = ops.sub(acc, ops.pow(ops.add(x, j), -1));
  }
  return acc;
}

export class ErfBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    return [ops.mul(gradOutputs[0], _erfDeriv(input.detach()))];
  }
}

export class ErfcBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    return [ops.neg(ops.mul(gradOutputs[0], _erfDeriv(input.detach())))];
  }
}

export class LgammaBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    return [ops.mul(gradOutputs[0], _digammaTensor(input.detach()))];
  }
}

export class GammaBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    return [ops.mul(gradOutputs[0], ops.mul(ops.gamma(x), _digammaTensor(x)))];
  }
}

export class ExpBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    return [ops.mul(gradOutputs[0], ops.exp(input.detach()))];
  }
}

export class LogBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    return [ops.div(gradOutputs[0], input.detach())];
  }
}

export class SqrtBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    const sqrtX = ops.sqrt(x);
    const two = full(x.shape, 2, { dtype: x.dtype, device: x.device });
    return [ops.div(gradOutputs[0], ops.mul(two, sqrtX))];
  }
}

export class TanhBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    const th = ops.tanh(x);
    const one = ones(x.shape, { dtype: x.dtype, device: x.device });
    return [ops.mul(gradOutputs[0], ops.sub(one, ops.mul(th, th)))];
  }
}

export class SigmoidBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    const sig = ops.sigmoid(x);
    const one = ones(x.shape, { dtype: x.dtype, device: x.device });
    return [ops.mul(gradOutputs[0], ops.mul(sig, ops.sub(one, sig)))];
  }
}

export class SoftmaxBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const args = this.opArgs();
    const dim = (args && args.length > 1 && args[1] != null) ? args[1] as number : -1;
    const g = gradOutputs[0];
    const y = ops.softmax(input.detach(), dim);
    const inner = ops.sum(ops.mul(g, y), dim, true);
    return [ops.mul(y, ops.sub(g, inner))];
  }
}

export class LogSoftmaxBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const args = this.opArgs();
    const dim = (args && args.length > 1 && args[1] != null) ? args[1] as number : -1;
    const g = gradOutputs[0];
    const sm = ops.softmax(input.detach(), dim);
    const gsum = ops.sum(g, dim, true);
    return [ops.sub(g, ops.mul(sm, gsum))];
  }
}

export class ReluBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
    const [input] = this.savedTensors();
    const x = input.detach();
    return [ops.mul(gradOutputs[0], ops.sign(ops.relu(x)))];
  }
}

export class GeluBackward extends AutogradNode {
  constructor() { super(1); }
  apply(gradOutputs: GradOutputList): GradInputList {
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
  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const [input] = this.savedTensors();
    const x = input.detach();
    const sig = ops.sigmoid(x);
    const one = ones(x.shape, { dtype: x.dtype, device: x.device });
    const grad = ops.mul(sig, ops.add(one, ops.mul(x, ops.sub(one, sig))));
    return [ops.mul(g, grad)];
  }
}
