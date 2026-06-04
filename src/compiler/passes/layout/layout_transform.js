import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, Layout } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';

export class LayoutTransformPass extends FunctionPass {
  constructor() {
    super('LayoutTransformPass');
  }

  run(func) {
    let changed = false;
    for (const op of [...func.ops()]) {
      if (op.opName === 'return' || op.opName === 'yield') continue;
      const def = registry.get(op.opName);
      if (!def || !def.isElementwise) continue;

      let targetLayout = null;
      for (let i = 0; i < op.numOperands; i++) {
        const t = op.getOperand(i).type;
        if (t instanceof TensorType && t.layout && !t.layout.isIdentity()) {
          targetLayout = t.layout;
          break;
        }
      }
      if (!targetLayout) continue;

      for (let i = 0; i < op.numOperands; i++) {
        const operand = op.getOperand(i);
        const t = operand.type;
        if (!(t instanceof TensorType)) continue;
        if (t.layout && t.layout.equals(targetLayout)) continue;
        if (t.layout && t.layout.isIdentity() && !targetLayout.isIdentity()) {
          const perm = targetLayout.order;
          const newShape = perm.map(j => t.shape[j]);
          const transposeOp = new Operation('transpose', [operand], [new TensorType(newShape, t.dtype, targetLayout)], { permutation: perm });
          if (op.parentBlock) {
            op.parentBlock.insertBefore(transposeOp, op);
            op.replaceOperand(i, transposeOp.getResult(0));
            changed = true;
          }
        }
      }
    }
    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
