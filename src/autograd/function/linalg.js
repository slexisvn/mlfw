import { AutogradNode } from '../node.js';
import * as ops from '../../tensor/ops/ops.js';
import { transpose, unsqueeze, squeeze } from '../../tensor/view/view_ops.js';

export class MatmulBackward extends AutogradNode {
  constructor() { super(2); }

  apply(gradOutputs) {
    const g = gradOutputs[0];
    const [self, other] = this.savedTensors();
    const a = self.detach();
    const b = other.detach();
    const aRank = a.ndim;
    const bRank = b.ndim;

    if (aRank === 2 && bRank === 2) {
      return [ops.matmul(g, transpose(b, 0, 1)), ops.matmul(transpose(a, 0, 1), g)];
    }

    if (aRank === 2 && bRank === 1) {
      const gUnsq = unsqueeze(g, 1);
      const bUnsq = unsqueeze(b, 0);
      return [ops.matmul(gUnsq, bUnsq), squeeze(ops.matmul(transpose(a, 0, 1), gUnsq), 1)];
    }

    if (aRank === 1 && bRank === 1) {
      return [ops.mul(g, b), ops.mul(g, a)];
    }

    if (aRank >= 3 && bRank >= 3) {
      return [
        ops.matmul(g, transpose(b, bRank - 2, bRank - 1)),
        ops.matmul(transpose(a, aRank - 2, aRank - 1), g),
      ];
    }

    throw new Error(`MatmulBackward: unsupported ranks ${aRank}, ${bRank}`);
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
