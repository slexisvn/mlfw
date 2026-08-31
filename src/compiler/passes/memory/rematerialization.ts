import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { derivedFrom } from '../../ir/graph/op_location.js';
import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';

import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import { LivenessAnalysis } from '../../analysis/liveness.js';
import { isConstantOp, isTerminatorOp } from '../../ir/graph/op_traits.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Block } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { AttrValue } from '../../ir/graph/types.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';

export type RematerializationOpts = {
  memoryBudget?: number;
  maxIterations?: number;
  maxRecomputeCost?: number;
  excludeOps?: ReadonlySet<string>;
};
type OpOrder = ReturnType<typeof LivenessAnalysis.buildIntervals>['opIndex'];
type RematCandidate = {
  value: Value;
  definingOp: Operation;
  memorySaved: number;
  recomputeCost: number;
  score: number;
};
type PressureAnalysis = { peakPressure: number; peakIdx: number; candidates: RematCandidate[]; opIndex: OpOrder };

export class RematerializationConfig {
  memoryBudget: number;
  maxIterations: number;
  maxRecomputeCost: number;
  excludeOps: ReadonlySet<string>;

  constructor(opts: RematerializationOpts = {}) {
    this.memoryBudget = opts.memoryBudget || Infinity;
    this.maxIterations = opts.maxIterations || 100;
    this.maxRecomputeCost = opts.maxRecomputeCost || Infinity;
    this.excludeOps = opts.excludeOps || new Set();
  }
}

export class RematerializationPass extends FunctionPass {
  config: RematerializationConfig;

  constructor(config: RematerializationOpts | RematerializationConfig = {}) {
    super('RematerializationPass');
    this.requiredAnalyses = [LivenessAnalysis];
    this.config = config instanceof RematerializationConfig
      ? config
      : new RematerializationConfig(config);
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    if (this.config.memoryBudget === Infinity) return PassResult.UNCHANGED;

    const explain = explainer(this.trace, this.name);
    let changed = false;
    let iterations = 0;

    let lastPeak = 0;

    while (iterations < this.config.maxIterations) {
      const { peakPressure, candidates, opIndex } = this._analyzeIntervalPressure(graphFunc, analysisManager);
      lastPeak = peakPressure;
      if (peakPressure <= this.config.memoryBudget) break;
      if (candidates.length === 0) break;

      candidates.sort((a, b) => b.score - a.score);
      const chosen = candidates[0];
      if (explain) {
        explain(chosen.definingOp.opName, 'recomputed instead of kept alive',
          'peak live memory is over budget, and of every value still live across the peak this one frees the most bytes per unit of recompute',
          { bytesFreed: chosen.memorySaved, recomputeCost: chosen.recomputeCost,
            peakPressure, budget: this.config.memoryBudget });
      }
      this._rematerialize(graphFunc, chosen, opIndex);
      changed = true;
      iterations++;
    }

    if (this.trace && lastPeak > this.config.memoryBudget) {
      this.trace.warn('rematerialization', graphFunc.name,
        `budget not met: peak live pressure is ${lastPeak} bytes against a budget of ${this.config.memoryBudget} bytes`,
        { peakPressure: lastPeak, budget: this.config.memoryBudget, iterations });
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        iterations, peakPressure: lastPeak,
        budget: this.config.memoryBudget, changed,
        level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _analyzeIntervalPressure(func: GraphFunction, analysisManager?: AnalysisManager): PressureAnalysis {
    const liveness = this.getAnalysis(LivenessAnalysis, func, analysisManager);
    const { peakPressure, peakIndex: peakIdx, liveAtPeak, opIndex } = liveness;

    const candidates: RematCandidate[] = [];
    for (const value of liveAtPeak) {
      if (!this._canRematerialize(value)) continue;
      const defOp = value.definingOp as Operation;
      const memorySaved = this._computeMemorySaved(value);
      if (memorySaved <= 0) continue;
      const recomputeCost = this._estimateRecomputeCost(defOp);
      if (recomputeCost >= this.config.maxRecomputeCost) continue;
      if (recomputeCost === 0) continue;
      candidates.push({
        value,
        definingOp: defOp,
        memorySaved,
        recomputeCost,
        score: memorySaved / recomputeCost
      });
    }

    return { peakPressure, peakIdx, candidates, opIndex };
  }

  _canRematerialize(value: Value): boolean {
    if (value.isBlockArgument()) return false;

    const defOp = value.definingOp;
    if (!defOp) return false;
    if (isTerminatorOp(defOp.opName) || isConstantOp(defOp.opName)) return false;
    if (this.config.excludeOps.has(defOp.opName)) return false;
    if (defOp.regions.length > 0) return false;
    if (defOp.hasSideEffects()) return false;
    if (value.useCount <= 1) return false;

    for (let i = 0; i < defOp.numOperands; i++) {
      const operand = defOp.getOperand(i);
      if (operand.definingOp && operand.definingOp.hasSideEffects()) return false;
    }

    return true;
  }

  _computeMemorySaved(value: Value): number {
    if (!(value.type instanceof TensorType)) return 0;
    const bytes = value.type.sizeInBytes() as number;
    if (bytes === DYNAMIC) return 0;
    return bytes;
  }

  _operandExtensionCost(defOp: Operation, cloneIdx: number, opOrder: OpOrder): number {
    let cost = 0;
    for (let i = 0; i < defOp.numOperands; i++) {
      const operand = defOp.getOperand(i);
      let maxUseIdx = 0;
      for (const use of operand.uses()) {
        const idx = opOrder.get(use.user) || 0;
        if (idx > maxUseIdx) maxUseIdx = idx;
      }
      if (cloneIdx > maxUseIdx) {
        if (operand.type instanceof TensorType) {
          const bytes = operand.type.sizeInBytes() as number;
          if (bytes !== DYNAMIC) cost += bytes;
        }
      }
    }
    return cost;
  }

  _estimateRecomputeCost(op: Operation): number {
    const def = registry.get(op.opName);
    if (!def) return Infinity;
    if (def.isOpaque) return Infinity;

    if (def.getFlops) {
      const flops = def.getFlops(op);
      if (flops > 0) return flops;
    }

    if (def.isElementwise || def.isBroadcast || def.isView) {
      let elements = 0;
      for (let i = 0; i < op.numResults; i++) {
        const t = op.getResult(i).type;
        if (t instanceof TensorType) {
          const n = t.numel() as number;
          if (n !== DYNAMIC) { elements = n; break; }
        }
      }
      return elements || 1;
    }

    if (def.isReduction) {
      for (let i = 0; i < op.numOperands; i++) {
        const t = op.getOperand(i).type;
        if (t instanceof TensorType) {
          const n = t.numel() as number;
          if (n !== DYNAMIC) return n;
        }
      }
    }

    return Infinity;
  }

  _rematerialize(func: GraphFunction, candidate: RematCandidate, opOrder: OpOrder): void {
    const { value, definingOp } = candidate;
    const uses: { user: Operation; operandIndex: number }[] = [];
    for (const use of value.uses()) {
      uses.push({ user: use.user, operandIndex: use.operandIndex });
    }

    if (uses.length <= 1) return;

    uses.sort((a, b) => (opOrder.get(a.user) || 0) - (opOrder.get(b.user) || 0));

    const lastUseIdx = opOrder.get(uses[uses.length - 1].user) || 0;
    const operandsCostIfExtended = this._operandExtensionCost(definingOp, lastUseIdx, opOrder);
    const savings = this._computeMemorySaved(value);
    if (operandsCostIfExtended >= savings) return;

    for (let i = 1; i < uses.length; i++) {
      const use = uses[i];
      const user = use.user;
      if (!user.parentBlock) continue;

      const clonedOp = derivedFrom(new Operation(
        definingOp.opName,
        [...definingOp.operands],
        definingOp.results.map(r => r.type),
        new Map<string, AttrValue>(definingOp.attributes)
      ), definingOp);

      (user.parentBlock as Block).insertBefore(clonedOp, user);
      user.replaceOperand(use.operandIndex, clonedOp.getResult(value.resultIndex));
    }
  }
}
