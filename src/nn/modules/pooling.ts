import { Module } from '../module.js';
import * as Fp from '../functional/pooling.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { Pool2dPadding, Pool2dSize, Pair2 } from '../functional/pooling.js';

export class MaxPool2d extends Module {
  kernelSize: Pair2;
  stride: Pair2;
  padding: Pool2dPadding;

  constructor(kernelSize: Pool2dSize, stride?: Pool2dSize | null, padding: Pool2dPadding = 0) {
    super();
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : this.kernelSize;
    this.padding = padding;
  }

  forward(input: Tensor): Tensor {
    return Fp.max_pool2d(input, this.kernelSize, this.stride, this.padding);
  }
}

export class AvgPool2d extends Module {
  kernelSize: Pair2;
  stride: Pair2;
  padding: Pool2dPadding;

  constructor(kernelSize: Pool2dSize, stride?: Pool2dSize | null, padding: Pool2dPadding = 0) {
    super();
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : this.kernelSize;
    this.padding = padding;
  }

  forward(input: Tensor): Tensor {
    return Fp.avg_pool2d(input, this.kernelSize, this.stride, this.padding);
  }
}

export class AdaptiveAvgPool2d extends Module {
  outputSize: Pair2;

  constructor(outputSize: Pool2dSize) {
    super();
    this.outputSize = Array.isArray(outputSize) ? outputSize : [outputSize, outputSize];
  }

  forward(input: Tensor): Tensor {
    return Fp.adaptive_avg_pool2d(input, this.outputSize);
  }
}
