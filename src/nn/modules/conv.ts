import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import * as Fc from '../functional/conv.js';
import { resetLinearParameters } from '../init.js';
import { zeros, empty } from '../../tensor/factory/creation_ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { ConvPadding2d, ConvSize2d, Pair2 } from '../functional/conv.js';

type Conv2dOptions = {
  stride?: ConvSize2d;
  padding?: ConvPadding2d;
  dilation?: ConvSize2d;
  groups?: number;
  bias?: boolean;
};

type Conv1dOptions = {
  stride?: number | readonly number[];
  padding?: number | Pair2;
  dilation?: number | readonly number[];
  groups?: number;
  bias?: boolean;
};

export class Conv2d extends Module {
  inChannels: number;
  outChannels: number;
  kernelSize: Pair2;
  stride: Pair2;
  padding: ConvPadding2d;
  dilation: Pair2;
  groups: number;
  weight: Parameter;
  bias: Parameter | null;

  constructor(inChannels: number, outChannels: number, kernelSize: ConvSize2d, opts: Conv2dOptions = {}) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = opts.stride ? (Array.isArray(opts.stride) ? opts.stride : [opts.stride, opts.stride]) : [1, 1];
    this.padding = opts.padding ?? 0;
    this.dilation = opts.dilation ? (Array.isArray(opts.dilation) ? opts.dilation : [opts.dilation, opts.dilation]) : [1, 1];
    this.groups = opts.groups ?? 1;

    const kH = this.kernelSize[0];
    const kW = this.kernelSize[1];
    this.weight = new Parameter(empty([outChannels, inChannels / this.groups, kH, kW]));
    this.bias = (opts.bias !== false) ? new Parameter(zeros([outChannels])) : null;
    this._resetParameters();
  }

  _resetParameters(): void {
    resetLinearParameters(this.weight, this.bias);
  }

  forward(input: Tensor): Tensor {
    return Fc.conv2d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}

export class Conv1d extends Module {
  inChannels: number;
  outChannels: number;
  kernelSize: number;
  stride: number | readonly number[];
  padding: number | Pair2;
  dilation: number | readonly number[];
  groups: number;
  weight: Parameter;
  bias: Parameter | null;

  constructor(inChannels: number, outChannels: number, kernelSize: number | readonly number[], opts: Conv1dOptions = {}) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
    this.stride = opts.stride ?? 1;
    this.padding = opts.padding ?? 0;
    this.dilation = opts.dilation ?? 1;
    this.groups = opts.groups ?? 1;

    this.weight = new Parameter(empty([outChannels, inChannels / this.groups, this.kernelSize]));
    this.bias = (opts.bias !== false) ? new Parameter(zeros([outChannels])) : null;
    resetLinearParameters(this.weight, this.bias);
  }

  forward(input: Tensor): Tensor {
    return Fc.conv1d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
