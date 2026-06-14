import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { full, zeros } from '../../tensor/factory/creation_ops.js';
import { unsqueeze } from '../../tensor/view/view_ops.js';

function _normalizeDims(dim, rank) {
  if (dim === undefined || dim === null) {
    const all = [];
    for (let i = 0; i < rank; i++) all.push(i);
    return all;
  }
  const list = Array.isArray(dim) ? dim : [dim];
  return list.map((d) => (d < 0 ? d + rank : d)).sort((a, b) => a - b);
}

function _unreduce(grad, inputShape, dims, keepdim) {
  let g = grad;
  if (!keepdim) {
    for (const d of dims) {
      g = unsqueeze(g, d);
    }
  }
  const z = zeros(inputShape, { dtype: g.dtype, device: g.device });
  return ops.add(z, g);
}

export class SumBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta.shape;
    const args = this.opArgs();
    const dim = args ? args[1] : undefined;
    const keepdim = args ? args[2] : false;
    const dims = _normalizeDims(dim, inputShape.length);
    return [_unreduce(g, inputShape, dims, keepdim)];
  }
}

export class MeanBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta.shape;
    const args = this.opArgs();
    const dim = args ? args[1] : undefined;
    const keepdim = args ? args[2] : false;
    const dims = _normalizeDims(dim, inputShape.length);

    let count = 1;
    for (const d of dims) count *= inputShape[d];

    const expanded = _unreduce(g, inputShape, dims, keepdim);
    const n = full(inputShape, count, { dtype: g.dtype, device: g.device });
    return [ops.div(expanded, n)];
  }
}
