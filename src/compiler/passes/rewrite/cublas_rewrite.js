import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { cublasDotInfo } from '../partition/cublas_split.js';

export class CublasRewritePass extends FunctionPass {
  constructor(config = {}) {
    super('CublasRewritePass');
    this.config = config;
  }

  run(func) {
    let changed = false;
    for (const op of [...func.ops()]) {
      if (op.opName !== 'dot') continue;
      if (op.numOperands !== 2) continue;
      if (cublasDotInfo(op) === null) continue;
      const block = op.parentBlock;
      if (!block) continue;

      const resultTypes = op.results.map(r => r.type);
      const gemm = new Operation('cublas_gemm', [op.getOperand(0), op.getOperand(1)], resultTypes, new Map(op.attributes), null);
      block.insertBefore(gemm, op);
      op.replaceAllResultsWith(gemm.results);
      op.erase();
      changed = true;
    }
    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
