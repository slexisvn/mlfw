import { Module } from '../module.js';
import { dropout } from '../functional/dropout.js';

export class Dropout extends Module {
  constructor(p = 0.5) {
    super();
    this.p = p;
  }

  forward(input) {
    return dropout(input, this.p, this.training);
  }
}
