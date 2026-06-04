import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder } from '../../ir/graph/builder.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';

export class ConstantFoldPass extends FunctionPass {
  constructor() {
    super('constant_fold');
  }

  run(func, analysisManager) {
    let changed = false;
    const builder = new IRBuilder(func);

    for (const op of [...func.opsArray()]) {
      if (!op.parentBlock) continue;
      if (op.opName === 'constant' || op.opName === 'scalar_constant') continue;

      const def = registry.get(op.opName);
      if (!def || op.regions.length > 0) continue;

      if (def.sideEffects || (def.hasTrait && def.hasTrait(OpTrait.SIDE_EFFECT))) continue;
      if (def.getMemoryEffects && def.getMemoryEffects(op).length > 0) continue;
      if (!def.fold) continue;

      let allConstants = true;
      const constValues = [];
      const constOps = [];
      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (!producer || (producer.opName !== 'constant' && producer.opName !== 'scalar_constant')) {
          allConstants = false;
          break;
        }
        constValues.push(producer.getAttr('value'));
        constOps.push(producer);
      }

      if (!allConstants || op.numOperands === 0) continue;

      try {
        const resultVal = def.fold(constValues, op.attributes, constOps);

        if (resultVal !== undefined) {
          builder.block = op.parentBlock;
          builder.setInsertionPoint(op);
          const newConst = builder.constant(resultVal, op.getResult(0).type);
          op.replaceAllResultsWith([newConst.getResult(0)]);
          op.erase();
          changed = true;
        }
      } catch (e) {
      }
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
