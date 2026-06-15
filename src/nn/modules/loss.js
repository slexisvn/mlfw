import { Module } from '../module.js';
import * as Fl from '../functional/loss.js';

export class MSELoss extends Module {
  constructor(reduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input, target) {
    return Fl.mse_loss(input, target, this.reduction);
  }
}

export class CrossEntropyLoss extends Module {
  constructor(reduction = 'mean', ignoreIndex = -100) {
    super();
    this.reduction = reduction;
    this.ignoreIndex = ignoreIndex;
  }

  forward(input, target) {
    return Fl.cross_entropy(input, target, this.reduction, this.ignoreIndex === -100 ? null : this.ignoreIndex);
  }
}

export class NLLLoss extends Module {
  constructor(reduction = 'mean', ignoreIndex = -100) {
    super();
    this.reduction = reduction;
    this.ignoreIndex = ignoreIndex;
  }

  forward(input, target) {
    return Fl.nll_loss(input, target, this.reduction, this.ignoreIndex === -100 ? null : this.ignoreIndex);
  }
}

export class BCELoss extends Module {
  constructor(reduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input, target) {
    return Fl.binary_cross_entropy(input, target, this.reduction);
  }
}
