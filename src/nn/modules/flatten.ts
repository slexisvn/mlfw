import { Module } from '../module.js';
import { reshape } from '../../tensor/ops/ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export class Flatten extends Module {
  startDim: number;
  endDim: number;

  constructor(startDim = 1, endDim = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }

  forward(input: Tensor): Tensor {
    const shape = input.shape;
    const ndim = shape.length;
    const start = this.startDim < 0 ? ndim + this.startDim : this.startDim;
    const end = this.endDim < 0 ? ndim + this.endDim : this.endDim;

    let flatSize = 1;
    for (let i = start; i <= end; i++) flatSize *= shape[i];

    const newShape: number[] = [];
    for (let i = 0; i < start; i++) newShape.push(shape[i]);
    newShape.push(flatSize);
    for (let i = end + 1; i < ndim; i++) newShape.push(shape[i]);

    return reshape(input, newShape);
  }
}
