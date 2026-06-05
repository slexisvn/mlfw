import { Module } from '../module.js';
import * as F from '../functional/activation.js';

export class ReLU extends Module {
  forward(input) { return F.relu(input); }
}

export class GELU extends Module {
  forward(input) { return F.gelu(input); }
}

export class SiLU extends Module {
  forward(input) { return F.silu(input); }
}

export class Sigmoid extends Module {
  forward(input) { return F.sigmoid(input); }
}

export class Tanh extends Module {
  forward(input) { return F.tanh(input); }
}

export class LeakyReLU extends Module {
  constructor(negativeSlope = 0.01) {
    super();
    this.negativeSlope = negativeSlope;
  }
  forward(input) { return F.leaky_relu(input, this.negativeSlope); }
}

export class ELU extends Module {
  constructor(alpha = 1.0) {
    super();
    this.alpha = alpha;
  }
  forward(input) { return F.elu(input, this.alpha); }
}

export class Softmax extends Module {
  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input) { return F.softmax(input, this.dim); }
}

export class LogSoftmax extends Module {
  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input) { return F.log_softmax(input, this.dim); }
}
