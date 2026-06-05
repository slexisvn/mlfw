import { Module } from '../module.js';
import * as Fp from '../functional/pooling.js';

export class MaxPool2d extends Module {
  constructor(kernelSize, stride, padding = 0) {
    super();
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : this.kernelSize;
    this.padding = padding;
  }

  forward(input) {
    return Fp.max_pool2d(input, this.kernelSize, this.stride, this.padding);
  }
}

export class AvgPool2d extends Module {
  constructor(kernelSize, stride, padding = 0) {
    super();
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : this.kernelSize;
    this.padding = padding;
  }

  forward(input) {
    return Fp.avg_pool2d(input, this.kernelSize, this.stride, this.padding);
  }
}

export class AdaptiveAvgPool2d extends Module {
  constructor(outputSize) {
    super();
    this.outputSize = Array.isArray(outputSize) ? outputSize : [outputSize, outputSize];
  }

  forward(input) {
    return Fp.adaptive_avg_pool2d(input, this.outputSize);
  }
}
