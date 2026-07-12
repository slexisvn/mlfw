import { Module } from '../module.js';
import { reshape } from '../../tensor/ops/ops.js';

export class Flatten extends Module {
  constructor(startDim = 1, endDim = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }

  forward(input) {
    const shape = input.shape;
    const ndim = shape.length;
    const start = this.startDim < 0 ? ndim + this.startDim : this.startDim;
    const end = this.endDim < 0 ? ndim + this.endDim : this.endDim;

    let flatSize = 1;
    for (let i = start; i <= end; i++) flatSize *= shape[i];

    const newShape = [];
    for (let i = 0; i < start; i++) newShape.push(shape[i]);
    newShape.push(flatSize);
    for (let i = end + 1; i < ndim; i++) newShape.push(shape[i]);

    return reshape(input, newShape);
  }
}
