import { FunctionPass, PassResult } from '../pass.js';
import { MemoryEffectAnalysis } from '../../analysis/memory_effect.js';
import { isTerminatorOp } from '../../ir/graph/op_traits.js';
import { explainer } from '../explain.js';

import { TraceLevel } from '../../support/trace.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { MemoryEffectResult } from '../../analysis/memory_effect.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export class DCEPass extends FunctionPass {
  constructor() {
    super('dce');
    this.requiredAnalyses = [MemoryEffectAnalysis];
    this.preservedAnalyses = new Set([MemoryEffectAnalysis as never]);
  }

  override run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const func = target as GraphFunction;
    let changed = false;

    const memEffects = this.getAnalysis(MemoryEffectAnalysis as never, func, analysisManager) as MemoryEffectResult;

    const explain = explainer(this.trace, this.name);
    const worklist: Operation[] = [];
    for (const op of func.opsRecursive()) {
      if (this._isDead(op, memEffects)) {
        worklist.push(op);
      }
    }

    const cascaded = new Set<Operation>();
    let erasedCount = 0;

    while (worklist.length > 0) {
      const op = worklist.pop() as Operation;
      if (!op.parentBlock) continue;
      if (!this._isDead(op, memEffects)) continue;

      const operandDefs: Operation[] = [];
      for (const consumed of this._valuesReadBy(op)) {
        const defOp = consumed.definingOp;
        if (defOp && defOp.parentBlock) operandDefs.push(defOp);
      }

      if (explain) {
        explain(op.opName, 'deleted', cascaded.has(op)
          ? 'its only reader was deleted a moment ago, so nothing reads it any more'
          : 'nothing reads its result and it declares no side effect');
      }

      this._eraseRecursively(op);
      changed = true;
      erasedCount++;

      for (const defOp of operandDefs) {
        if (defOp.parentBlock && this._isDead(defOp, memEffects)) {
          cascaded.add(defOp);
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

  _valuesReadBy(op: Operation): Value[] {
    const values: Value[] = [];
    const visit = (current: Operation): void => {
      for (let i = 0; i < current.numOperands; i++) values.push(current.getOperand(i));
      for (const region of current.regions) {
        for (const block of region.blocks) {
          for (const inner of block.ops()) visit(inner);
        }
      }
    };
    visit(op);
    return values;
  }

  _isDead(op: Operation, memEffects: MemoryEffectResult): boolean {
    if (isTerminatorOp(op.opName)) return false;

    for (let i = 0; i < op.numResults; i++) {
      if (op.getResult(i).hasUses) return false;
    }

    return !memEffects.hasSideEffect(op);
  }

  _eraseRecursively(op: Operation): void {
    const nested: Operation[] = [];
    const collect = (current: Operation): void => {
      for (const region of current.regions) {
        for (const block of region.blocks) {
          for (const inner of block.opsArray()) {
            nested.push(inner);
            collect(inner);
          }
        }
      }
    };
    collect(op);

    for (const inner of nested) inner.dropAllOperands();
    for (let i = nested.length - 1; i >= 0; i--) nested[i].erase();
    op.erase();
  }
}
