import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import * as Fn from '../functional/normalization.js';
import { ones, zeros } from '../../tensor/factory/creation_ops.js';

export class LayerNorm extends Module {
  constructor(normalizedShape, eps = 1e-5, elementwiseAffine = true) {
    super();
    this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
    this.eps = eps;
    this.weight = elementwiseAffine ? new Parameter(ones(this.normalizedShape)) : null;
    this.bias = elementwiseAffine ? new Parameter(zeros(this.normalizedShape)) : null;
  }

  forward(input) {
    return Fn.layer_norm(input, this.normalizedShape, this.weight, this.bias, this.eps);
  }
}

export class BatchNorm1d extends Module {
  constructor(numFeatures, eps = 1e-5, affine = true) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.weight = affine ? new Parameter(ones([numFeatures])) : null;
    this.bias = affine ? new Parameter(zeros([numFeatures])) : null;
    this.runningMean = zeros([numFeatures]);
    this.runningVar = ones([numFeatures]);
    this.registerBuffer('runningMean', this.runningMean);
    this.registerBuffer('runningVar', this.runningVar);
  }

  forward(input) {
    return Fn.batch_norm(input, this.runningMean, this.runningVar, this.weight, this.bias, this.training, this.eps);
  }
}

export class BatchNorm2d extends BatchNorm1d {
  constructor(numFeatures, eps = 1e-5, affine = true) {
    super(numFeatures, eps, affine);
  }
}
