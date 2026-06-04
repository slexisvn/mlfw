import { FunctionPass, PassResult } from '../pass.js';
import { MemoryEffectAnalysis } from '../../analysis/memory_effect.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { ShapeAnalysis } from '../../analysis/shape_analysis.js';

export class DCEPass extends FunctionPass {
  constructor() {
    super('dce');
    this.preservedAnalyses = new Set([ShapeAnalysis]);
  }

  run(func, analysisManager) {
    let changed = false;

    const memEffects = analysisManager
      ? analysisManager.getAnalysis(MemoryEffectAnalysis, func)
      : MemoryEffectAnalysis.compute(func);

    const worklist = [];
    for (const op of func.opsArray()) {
      if (this._isDead(op, memEffects)) {
        worklist.push(op);
      }
    }

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

      for (const defOp of operandDefs) {
        if (defOp.parentBlock && this._isDead(defOp, memEffects)) {
          worklist.push(defOp);
        }
      }
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _isDead(op, memEffects) {
    if (op.opName === 'return' || op.opName === 'yield') return false;

    for (let i = 0; i < op.numResults; i++) {
      if (op.getResult(i).hasUses) return false;
    }

    return !memEffects.hasSideEffect(op);
  }
}
