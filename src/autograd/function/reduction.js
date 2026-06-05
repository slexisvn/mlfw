import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { ones, full, zeros } from '../../tensor/factory/creation_ops.js';
import { expand, unsqueeze, reshape } from '../../tensor/view/view_ops.js';

export class SumBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta.shape;
    return [_expandToShape(g, inputShape)];
  }
}

export class MeanBackward extends AutogradNode {
  constructor() { super(1); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const meta = this.inputMetadata(0);
    const inputShape = meta.shape;
    let numel = 1;
    for (let i = 0; i < inputShape.length; i++) numel *= inputShape[i];

    const expanded = _expandToShape(g, inputShape);
    const n = full(inputShape, numel, { dtype: g.dtype, device: g.device });
    return [ops.div(expanded, n)];
  }
}

function _expandToShape(grad, targetShape) {
  if (grad.ndim === 0) {
    return full(targetShape, grad.item(), { dtype: grad.dtype, device: grad.device });
  }

  let g = grad;
  while (g.ndim < targetShape.length) {
    g = unsqueeze(g, 0);
  }

  return expand(g, targetShape);
}
