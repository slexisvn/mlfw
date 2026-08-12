import { FunctionPass, PassResult } from '../pass.js';
import { MemoryEffectAnalysis } from '../../analysis/memory_effect.js';
import { isTerminatorOp } from '../../ir/graph/op_traits.js';

import { TraceLevel } from '../../pipeline/trace.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { MemoryEffectResult } from '../../analysis/memory_effect.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export class DCEPass extends FunctionPass {
  constructor() {
    super('dce');
    this.preservedAnalyses = new Set();
    this.requiredAnalyses = [MemoryEffectAnalysis];
  }

  override run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const func = target as GraphFunction;
    let changed = false;

    const memEffects = (analysisManager
      ? analysisManager.getAnalysis(MemoryEffectAnalysis as never, func)
      : MemoryEffectAnalysis.compute(func)) as MemoryEffectResult;

    const worklist: Operation[] = [];
    for (const op of func.opsRecursive()) {
      if (this._isDead(op, memEffects)) {
        worklist.push(op);
      }
    }

    let erasedCount = 0;

    while (worklist.length > 0) {
      const op = worklist.pop() as Operation;
      if (!op.parentBlock) continue;
      if (!this._isDead(op, memEffects)) continue;

      const operandDefs: Operation[] = [];
      for (let i = 0; i < op.numOperands; i++) {
        const defOp = op.getOperand(i).definingOp;
        if (defOp && defOp.parentBlock) operandDefs.push(defOp);
      }

      op.erase();
      changed = true;
      erasedCount++;

      for (const defOp of operandDefs) {
        if (defOp.parentBlock && this._isDead(defOp, memEffects)) {
          worklist.push(defOp);
        }
      }
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG && erasedCount > 0) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        erasedCount, level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _isDead(op: Operation, memEffects: MemoryEffectResult): boolean {
    if (isTerminatorOp(op.opName)) return false;

    if (op.regions && op.regions.length > 0) return false;

    for (let i = 0; i < op.numResults; i++) {
      if (op.getResult(i).hasUses) return false;
    }

    return !memEffects.hasSideEffect(op);
  }
}
