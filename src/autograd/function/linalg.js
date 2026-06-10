import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { transpose, unsqueeze, squeeze, reshape } from '../../tensor/view/view_ops.js';

function _sumToShape(tensor, targetShape) {
  let t = tensor;
  while (t.ndim > targetShape.length) {
    t = ops.sum(t, 0, false);
  }
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && t.shape[i] !== 1) {
      t = ops.sum(t, i, true);
    }
  }
  return t;
}

export class MatmulBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [self, other] = this.savedTensors();
    const a = self.detach();
    const b = other.detach();
    const aRank = a.ndim;
    const bRank = b.ndim;

    if (aRank === 1 && bRank === 1) {
      return [ops.mul(g, b), ops.mul(g, a)];
    }

    if (aRank === 2 && bRank === 1) {
      const gUnsq = unsqueeze(g, 1);
      const bUnsq = unsqueeze(b, 0);
      return [ops.matmul(gUnsq, bUnsq), squeeze(ops.matmul(transpose(a, 0, 1), gUnsq), 1)];
    }

    if (aRank === 1 && bRank === 2) {
      const gradA = ops.matmul(g, transpose(b, 0, 1));
      const gradB = ops.matmul(unsqueeze(a, 1), unsqueeze(g, 0));
      return [gradA, gradB];
    }

    let gradA = ops.matmul(g, transpose(b, bRank - 2, bRank - 1));
    let gradB = ops.matmul(transpose(a, aRank - 2, aRank - 1), g);

    gradA = _sumToShape(gradA, a.shape);
    gradB = _sumToShape(gradB, b.shape);

    return [gradA, gradB];
  }
}

export class DotBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [self, other] = this.savedTensors();
    return [ops.mul(g, other.detach()), ops.mul(g, self.detach())];
  }
}
