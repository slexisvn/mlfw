import { Module } from '../module.js';
import { dropout } from '../functional/dropout.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export class Dropout extends Module {
  p: number;

  constructor(p = 0.5) {
    super();
    this.p = p;
  }

  forward(input: Tensor): Tensor {
    return dropout(input, this.p, this.training);
  }
}
