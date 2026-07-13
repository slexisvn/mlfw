import { Module } from '../module.js';
import * as F from '../functional/activation.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export class ReLU extends Module {
  forward(input: Tensor): Tensor { return F.relu(input); }
}

export class GELU extends Module {
  forward(input: Tensor): Tensor { return F.gelu(input); }
}

export class SiLU extends Module {
  forward(input: Tensor): Tensor { return F.silu(input); }
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor { return F.sigmoid(input); }
}

export class Tanh extends Module {
  forward(input: Tensor): Tensor { return F.tanh(input); }
}

export class LeakyReLU extends Module {
  negativeSlope: number;

  constructor(negativeSlope = 0.01) {
    super();
    this.negativeSlope = negativeSlope;
  }
  forward(input: Tensor): Tensor { return F.leaky_relu(input, this.negativeSlope); }
}

export class ELU extends Module {
  alpha: number;

  constructor(alpha = 1.0) {
    super();
    this.alpha = alpha;
  }
  forward(input: Tensor): Tensor { return F.elu(input, this.alpha); }
}

export class Softmax extends Module {
  dim: number;

  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input: Tensor): Tensor { return F.softmax(input, this.dim); }
}

export class LogSoftmax extends Module {
  dim: number;

  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input: Tensor): Tensor { return F.log_softmax(input, this.dim); }
}
