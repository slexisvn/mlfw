import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { zeros } from '../../tensor/factory/creation_ops.js';
import { reshape, transpose, permute } from '../../tensor/view/view_ops.js';

export class ReshapeBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const meta = this.inputMetadata(0);
    return [reshape(gradOutputs[0], meta.shape)];
  }
}

export class TransposeBackward extends AutogradNode {
  constructor(dim0, dim1) {
    super(1);
    this._dim0 = dim0;
    this._dim1 = dim1;
  }

  apply(gradOutputs) {
    return [transpose(gradOutputs[0], this._dim0, this._dim1)];
  }
}

export class SliceBackward extends AutogradNode {
  constructor(dim, start, end, step) {
    super(1);
    this._dim = dim;
    this._start = start;
    this._end = end;
    this._step = step;
  }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const result = zeros(meta.shape, { dtype: g.dtype, device: g.device });

    const outData = result._impl.storage.data;
    const gData = g._impl.storage.data;
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
        const idx = d === this._dim
          ? this._start + indices[d] * (this._step || 1)
          : indices[d];
        oi += idx * resultStrides[d];
      }
      outData[oi] += gData[gi];

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
  constructor(dim, index) {
    super(1);
    this._dim = dim;
    this._index = index;
  }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const result = zeros(meta.shape, { dtype: g.dtype, device: g.device });

    const outData = result._impl.storage.data;
    const gData = g._impl.storage.data;
    const gOff = g._impl.storageOffset;
    const gShape = g.shape;
    const gStrides = g.strides;
    const resultStrides = result.strides;

    const gndim = gShape.length;
    const indices = new Int32Array(gndim);
    let gi = gOff;

    for (let i = 0; i < g.numel; i++) {
      let oi = this._index * resultStrides[this._dim];
      for (let d = 0; d < gndim; d++) {
        const od = d < this._dim ? d : d + 1;
        oi += indices[d] * resultStrides[od];
      }
      outData[oi] += gData[gi];

      for (let d = gndim - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < gShape[d]) { gi += gStrides[d]; break; }
        gi -= (gShape[d] - 1) * gStrides[d];
        indices[d] = 0;
      }
    }

    return [result];
  }
}

export class ExpandBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta.shape;
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
  constructor(dims) {
    super(1);
    this._dims = dims;
  }

  apply(gradOutputs) {
    const inversePerm = new Array(this._dims.length);
    for (let i = 0; i < this._dims.length; i++) {
      inversePerm[this._dims[i]] = i;
    }
    return [permute(gradOutputs[0], inversePerm)];
  }
}
