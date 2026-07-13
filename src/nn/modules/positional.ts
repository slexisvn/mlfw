import { Module } from '../module.js';
import { Dropout } from './dropout.js';
import * as ops from '../../tensor/ops/ops.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

type NarrowableTensor = Tensor & { narrow: (dim: number, start: number, length: number) => Tensor };

export class PositionalEncoding extends Module {
  dropoutLayer: Dropout;
  pe: Tensor;

  constructor(dModel: number, maxLen = 5000, dropout = 0.1) {
    super();
    this.dropoutLayer = new Dropout(dropout);
    const pe = zeros([1, maxLen, dModel]);
    const data = pe._impl.storage.data!;
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel; i += 2) {
        const angle = pos * Math.exp(-(i * Math.log(10000)) / dModel);
        data[pos * dModel + i] = Math.sin(angle);
        if (i + 1 < dModel) {
          data[pos * dModel + i + 1] = Math.cos(angle);
        }
      }
    }
    this.pe = pe;
    this.registerBuffer('pe', pe);
  }

  forward(x: Tensor): Tensor {
    const seqLen = x.shape[1];
    const peSlice = (this.pe as NarrowableTensor).narrow(1, 0, seqLen);
    return this.dropoutLayer.forward(ops.add(x, peSlice));
  }
}
