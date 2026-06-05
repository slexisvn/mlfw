import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import { linear } from '../functional/linear.js';
import { kaiming_uniform_, uniform_, _calculateFanInFanOut } from '../init.js';
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
    kaiming_uniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const { fanIn } = _calculateFanInFanOut(this.weight);
      const bound = 1 / Math.sqrt(fanIn);
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input) {
    return linear(input, this.weight, this.bias);
  }
}
