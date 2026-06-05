import { Module } from '../module.js';
import { Parameter } from '../parameter.js';
import * as Fc from '../functional/conv.js';
import { kaiming_uniform_, uniform_, _calculateFanInFanOut } from '../init.js';
import { zeros, empty } from '../../tensor/factory/creation_ops.js';

export class Conv2d extends Module {
  constructor(inChannels, outChannels, kernelSize, opts = {}) {
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

  _resetParameters() {
    kaiming_uniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const { fanIn } = _calculateFanInFanOut(this.weight);
      const bound = 1 / Math.sqrt(fanIn);
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input) {
    return Fc.conv2d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}

export class Conv1d extends Module {
  constructor(inChannels, outChannels, kernelSize, opts = {}) {
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
    kaiming_uniform_(this.weight, Math.sqrt(5));
  }

  forward(input) {
    return Fc.conv1d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
