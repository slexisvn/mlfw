import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import { linear } from '../functional/linear.js';
import { resetLinearParameters } from '../init.js';
import { zeros, empty } from '../../tensor/factory/creation_ops.js';

export class Linear extends Module {
  constructor(inFeatures, outFeatures, bias = true) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weight = new Parameter(empty([outFeatures, inFeatures]));
    this.bias = bias ? new Parameter(zeros([outFeatures])) : null;
    this._resetParameters();
  }

  _resetParameters() {
    resetLinearParameters(this.weight, this.bias);
  }

  forward(input) {
    return linear(input, this.weight, this.bias);
  }
}
