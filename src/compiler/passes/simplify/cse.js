import { FunctionPass, PassResult } from '../pass.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { ShapeAnalysis } from '../../analysis/shape_analysis.js';
import { TraceLevel } from '../../pipeline/trace.js';

export class CSEPass extends FunctionPass {
  constructor() {
    super('cse');
    this.preservedAnalyses = new Set([ShapeAnalysis]);
  }

  run(func, analysisManager) {
    let changed = false;
    let eliminated = 0;

    const blocks = typeof func.blocksRecursive === 'function'
      ? [...func.blocksRecursive()]
      : [...func.body];

    for (const block of blocks) {
      const available = new Map();
      for (const op of [...block.ops()]) {
        if (!op.parentBlock) continue;

        if (op.regions && op.regions.length > 0) continue;

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
            eliminated++;
            replaced = true;
            break;
          }
        }

        if (!replaced) {
          candidates.push(op);
        }
      }
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG && eliminated > 0) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        eliminated, level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
}
