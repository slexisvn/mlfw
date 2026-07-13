import { Module } from '../module.js';
import * as Fl from '../functional/loss.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { LossReduction } from '../functional/loss.js';

export class MSELoss extends Module {
  reduction: LossReduction;

  constructor(reduction: LossReduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return Fl.mse_loss(input, target, this.reduction);
  }
}

export class CrossEntropyLoss extends Module {
  reduction: LossReduction;
  ignoreIndex: number;

  constructor(reduction: LossReduction = 'mean', ignoreIndex = -100) {
    super();
    this.reduction = reduction;
    this.ignoreIndex = ignoreIndex;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return Fl.cross_entropy(input, target, this.reduction, this.ignoreIndex === -100 ? null : this.ignoreIndex);
  }
}

export class NLLLoss extends Module {
  reduction: LossReduction;
  ignoreIndex: number;

  constructor(reduction: LossReduction = 'mean', ignoreIndex = -100) {
    super();
    this.reduction = reduction;
    this.ignoreIndex = ignoreIndex;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return Fl.nll_loss(input, target, this.reduction, this.ignoreIndex === -100 ? null : this.ignoreIndex);
  }
}

export class BCELoss extends Module {
  reduction: LossReduction;

  constructor(reduction: LossReduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return Fl.binary_cross_entropy(input, target, this.reduction);
  }
}
