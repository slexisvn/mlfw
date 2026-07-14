import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import { linear } from '../functional/linear.js';
import { resetLinearParameters } from '../init.js';
import { zeros, empty } from '../../tensor/factory/creation_ops.js';
import type { NNTensor } from '../types.js';

export class Linear extends Module {
  inFeatures: number;
  outFeatures: number;
  weight: Parameter;
  bias: Parameter | null;

  constructor(inFeatures: number, outFeatures: number, bias = true) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weight = new Parameter(empty([outFeatures, inFeatures]));
    this.bias = bias ? new Parameter(zeros([outFeatures])) : null;
    this._resetParameters();
  }

  _resetParameters(): void {
    resetLinearParameters(this.weight, this.bias);
  }

  forward(input: NNTensor): NNTensor {
    return linear(input, this.weight as unknown as NNTensor, this.bias as unknown as NNTensor | null);
  }
}
