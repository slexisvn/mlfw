import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import * as Fn from '../functional/normalization.js';
import { ones, zeros } from '../../tensor/factory/creation_ops.js';
import type { NNTensor } from '../types.js';

export class LayerNorm extends Module {
  normalizedShape: number[];
  eps: number;
  weight: Parameter | null;
  bias: Parameter | null;

  constructor(normalizedShape: number | readonly number[], eps = 1e-5, elementwiseAffine = true) {
    super();
    this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
    this.eps = eps;
    this.weight = elementwiseAffine ? new Parameter(ones(this.normalizedShape)) : null;
    this.bias = elementwiseAffine ? new Parameter(zeros(this.normalizedShape)) : null;
  }

  forward(input: NNTensor): NNTensor {
    return Fn.layer_norm(input, this.normalizedShape, this.weight as unknown as NNTensor | null, this.bias as unknown as NNTensor | null, this.eps);
  }
}

export class GroupNorm extends Module {
  numGroups: number;
  numChannels: number;
  eps: number;
  weight: Parameter | null;
  bias: Parameter | null;

  constructor(numGroups: number, numChannels: number, eps = 1e-5, affine = true) {
    super();
    this.numGroups = numGroups;
    this.numChannels = numChannels;
    this.eps = eps;
    this.weight = affine ? new Parameter(ones([numChannels])) : null;
    this.bias = affine ? new Parameter(zeros([numChannels])) : null;
  }

  forward(input: NNTensor): NNTensor {
    return Fn.group_norm(input, this.numGroups, this.weight as unknown as NNTensor | null, this.bias as unknown as NNTensor | null, this.eps);
  }
}

export class BatchNorm1d extends Module {
  numFeatures: number;
  eps: number;
  weight: Parameter | null;
  bias: Parameter | null;
  runningMean: NNTensor;
  runningVar: NNTensor;

  constructor(numFeatures: number, eps = 1e-5, affine = true) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.weight = affine ? new Parameter(ones([numFeatures])) : null;
    this.bias = affine ? new Parameter(zeros([numFeatures])) : null;
    this.runningMean = zeros([numFeatures]) as NNTensor;
    this.runningVar = ones([numFeatures]) as NNTensor;
    this.registerBuffer('runningMean', this.runningMean);
    this.registerBuffer('runningVar', this.runningVar);
  }

  forward(input: NNTensor): NNTensor {
    return Fn.batch_norm(input, this.runningMean, this.runningVar, this.weight as unknown as NNTensor | null, this.bias as unknown as NNTensor | null, this.training, this.eps);
  }
}

export class BatchNorm2d extends BatchNorm1d {
  constructor(numFeatures: number, eps = 1e-5, affine = true) {
    super(numFeatures, eps, affine);
  }
}


export class RMSNorm extends Module {
  normalizedShape: readonly number[];
  eps: number;
  weight: Parameter | null;

  constructor(normalizedShape: number | readonly number[], eps = 1e-6, affine = true) {
    super();
    this.normalizedShape = typeof normalizedShape === 'number' ? [normalizedShape] : [...normalizedShape];
    this.eps = eps;
    this.weight = affine ? new Parameter(ones([...this.normalizedShape])) : null;
  }

  forward(input: NNTensor): NNTensor {
    return Fn.rms_norm(input, this.normalizedShape, this.weight as unknown as NNTensor | null, this.eps);
  }
}

export class InstanceNorm2d extends Module {
  numChannels: number;
  eps: number;
  weight: Parameter | null;
  bias: Parameter | null;

  constructor(numChannels: number, eps = 1e-5, affine = false) {
    super();
    this.numChannels = numChannels;
    this.eps = eps;
    this.weight = affine ? new Parameter(ones([numChannels])) : null;
    this.bias = affine ? new Parameter(zeros([numChannels])) : null;
  }

  forward(input: NNTensor): NNTensor {
    return Fn.instance_norm(input, this.weight as unknown as NNTensor | null, this.bias as unknown as NNTensor | null, this.eps);
  }
}

export class InstanceNorm1d extends InstanceNorm2d {}
