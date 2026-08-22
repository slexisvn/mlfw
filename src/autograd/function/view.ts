import { AutogradNode } from '../node.js';
import { addAt } from '../../tensor/utils/typed_array.js';
import * as ops from '../../tensor/ops/ops.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { reshape, transpose, permute, unsqueeze } from '../../tensor/ops/ops.js';
import type { GradInputList, GradOutputList } from '../types.js';

export class ReshapeBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const meta = this.inputMetadata(0);
    return [reshape(gradOutputs[0], meta!.shape)];
  }
}

export class TransposeBackward extends AutogradNode {
  private readonly _dim0: number;
  private readonly _dim1: number;

  constructor(dim0: number, dim1: number) {
    super(1);
    this._dim0 = dim0;
    this._dim1 = dim1;
  }

  apply(gradOutputs: GradOutputList): GradInputList {
    return [transpose(gradOutputs[0], this._dim0, this._dim1)];
  }
}

export class SliceBackward extends AutogradNode {
  private readonly _dim: number;
  private readonly _start: number;
  private readonly _end: number;
  private readonly _step: number;

  constructor(dim: number, start: number, end: number, step: number) {
    super(1);
    this._dim = dim;
    this._start = start;
    this._end = end;
    this._step = step;
  }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const dim = this._dim;
    const size = meta!.shape[dim];
    const step = this._step || 1;
    let start = this._start < 0 ? this._start + size : this._start;
    let end = this._end < 0 ? this._end + size : this._end;
    start = Math.max(0, Math.min(start, size));
    end = Math.max(0, Math.min(end, size));

    if (step === 1) {
      const low = meta!.shape.map(() => 0);
      const high = meta!.shape.map(() => 0);
      low[dim] = start;
      high[dim] = size - end;
      return [ops.pad(g, low, high, 0)];
    }

    const result = zeros(meta!.shape, { dtype: g.dtype, device: g.device });
    const outData = result._impl.storage.data!;
    const gData = g._impl.storage.data!;
    const gOff = g._impl.storageOffset;
    const gShape = g.shape;
    const gStrides = g.strides;
    const resultStrides = result.strides;
    const ndim = gShape.length;
    const indices = new Int32Array(ndim);
    let gi = gOff;

    for (let i = 0; i < g.numel; i++) {
      let oi = 0;
      for (let d = 0; d < ndim; d++) {
        const idx = d === dim ? start + indices[d] * step : indices[d];
        oi += idx * resultStrides[d];
      }
      addAt(outData, oi, gData[gi]);

      for (let d = ndim - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < gShape[d]) { gi += gStrides[d]; break; }
        gi -= (gShape[d] - 1) * gStrides[d];
        indices[d] = 0;
      }
    }

    return [result];
  }
}

export class SelectBackward extends AutogradNode {
  private readonly _dim: number;
  private readonly _index: number;

  constructor(dim: number, index: number) {
    super(1);
    this._dim = dim;
    this._index = index;
  }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const dim = this._dim;
    const size = meta!.shape[dim];
    const index = this._index < 0 ? this._index + size : this._index;
    const expanded = unsqueeze(g, dim);
    const low = meta!.shape.map(() => 0);
    const high = meta!.shape.map(() => 0);
    low[dim] = index;
    high[dim] = size - 1 - index;
    return [ops.pad(expanded, low, high, 0)];
  }
}

export class ExpandBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs: GradOutputList): GradInputList {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta!.shape;
    const gradShape = g.shape;

    const dimsToReduce = [];
    const dimDiff = gradShape.length - inputShape.length;
    for (let i = 0; i < dimDiff; i++) dimsToReduce.push(i);
    for (let i = 0; i < inputShape.length; i++) {
      if (inputShape[i] === 1 && gradShape[i + dimDiff] !== 1) {
        dimsToReduce.push(i + dimDiff);
      }
    }

    let result = g;
    if (dimsToReduce.length > 0) {
      result = ops.sum(g, dimsToReduce, true);
    }

    return [reshape(result, inputShape)];
  }
}

export class PermuteBackward extends AutogradNode {
  private readonly _dims: readonly number[];

  constructor(dims: readonly number[]) {
    super(1);
    this._dims = dims;
  }

  apply(gradOutputs: GradOutputList): GradInputList {
    const rank = this._dims.length;
    const inversePerm = new Array<number>(rank);
    for (let i = 0; i < rank; i++) {
      const d = this._dims[i] < 0 ? rank + this._dims[i] : this._dims[i];
      inversePerm[d] = i;
    }
    return [permute(gradOutputs[0], inversePerm)];
  }
}
