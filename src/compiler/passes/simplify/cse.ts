import { FunctionPass, PassResult } from '../pass.js';
import { registry } from '../../ir/graph/ops.js';
import { TraceLevel } from '../../pipeline/trace.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export class CSEPass extends FunctionPass {
  constructor() {
    super('cse');
    this.preservedAnalyses = new Set();
  }

  override run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const func = target as GraphFunction;
    let changed = false;
    let eliminated = 0;

    const blocks = typeof func.blocksRecursive === 'function'
      ? [...func.blocksRecursive()]
      : [...func.body];

    for (const block of blocks) {
      const available = new Map<number, Operation[]>();
      for (const op of [...block.ops()]) {
        if (!op.parentBlock) continue;

        if (op.regions && op.regions.length > 0) continue;

        const def = registry.get(op.opName);

        if (def && def.hasSideEffects) continue;
        if (def && def.getMemoryEffects && def.getMemoryEffects(op).length > 0) continue;

        const hash = op.structuralHash();

        if (!available.has(hash)) {
          available.set(hash, [op]);
          continue;
        }

        const candidates = available.get(hash) as Operation[];
        let replaced = false;

        for (const candidate of candidates) {
          if (!candidate.parentBlock) continue;
          if (candidate.structuralEquals(op)) {
            const results: Value[] = [];
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
