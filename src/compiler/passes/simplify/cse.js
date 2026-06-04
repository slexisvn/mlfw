import { FunctionPass, PassResult } from '../pass.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { ShapeAnalysis } from '../../analysis/shape_analysis.js';

export class CSEPass extends FunctionPass {
  constructor() {
    super('cse');
    this.preservedAnalyses = new Set([ShapeAnalysis]);
  }

  run(func, analysisManager) {
    let changed = false;
    const available = new Map();

    for (const op of [...func.opsArray()]) {
      if (!op.parentBlock) continue;

      const def = registry.get(op.opName);

      if (def && (def.sideEffects || (def.hasTrait && def.hasTrait(OpTrait.SIDE_EFFECT)))) continue;
      if (def && def.getMemoryEffects && def.getMemoryEffects(op).length > 0) continue;

      const hash = op.structuralHash();

      if (!available.has(hash)) {
        available.set(hash, [op]);
        continue;
      }

      const candidates = available.get(hash);
      let replaced = false;

      for (const candidate of candidates) {
        if (!candidate.parentBlock) continue;
        if (candidate.structuralEquals(op)) {
          const results = [];
          for (let i = 0; i < candidate.numResults; i++) {
            results.push(candidate.getResult(i));
          }
          op.replaceAllResultsWith(results);
          op.erase();
          changed = true;
          replaced = true;
          break;
        }
      }

      if (!replaced) {
        candidates.push(op);
      }
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
