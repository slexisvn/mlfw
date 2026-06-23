import { FunctionPass, PassResult } from '../pass.js';
import { MemoryEffectAnalysis } from '../../analysis/memory_effect.js';

import { ShapeAnalysis } from '../../analysis/shape_analysis.js';
import { TraceLevel } from '../../pipeline/trace.js';

export class DCEPass extends FunctionPass {
  constructor() {
    super('dce');
    this.preservedAnalyses = new Set([ShapeAnalysis]);
    this.requiredAnalyses = [MemoryEffectAnalysis];
  }

  run(func, analysisManager) {
    let changed = false;

    const memEffects = analysisManager
      ? analysisManager.getAnalysis(MemoryEffectAnalysis, func)
      : MemoryEffectAnalysis.compute(func);

    const worklist = [];
    for (const op of func.opsRecursive()) {
      if (this._isDead(op, memEffects)) {
        worklist.push(op);
      }
    }

    let erasedCount = 0;

    while (worklist.length > 0) {
      const op = worklist.pop();
      if (!op.parentBlock) continue;
      if (!this._isDead(op, memEffects)) continue;

      const operandDefs = [];
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

  _isDead(op, memEffects) {
    if (op.opName === 'return' || op.opName === 'yield') return false;

    if (op.regions && op.regions.length > 0) return false;

    for (let i = 0; i < op.numResults; i++) {
      if (op.getResult(i).hasUses) return false;
    }

    return !memEffects.hasSideEffect(op);
  }
}
