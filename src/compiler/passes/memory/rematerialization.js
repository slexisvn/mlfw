import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { LivenessAnalysis } from '../../analysis/liveness.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';

const NEVER_REMAT = new Set(['return', 'yield', 'constant', 'scalar_constant']);

export class RematerializationConfig {
  constructor(opts = {}) {
    this.memoryBudget = opts.memoryBudget || Infinity;
    this.maxIterations = opts.maxIterations || 100;
    this.maxRecomputeCost = opts.maxRecomputeCost || Infinity;
    this.excludeOps = opts.excludeOps || new Set();
  }
}

export class RematerializationPass extends FunctionPass {
  constructor(config = {}) {
    super('RematerializationPass');
    this.config = config instanceof RematerializationConfig
      ? config
      : new RematerializationConfig(config);
  }

  run(func, analysisManager) {
    if (this.config.memoryBudget === Infinity) return PassResult.UNCHANGED;

    let changed = false;
    let iterations = 0;

    while (iterations < this.config.maxIterations) {
      const useDef = UseDefAnalysis.compute(func);
      const { peakPressure, candidates } = this._analyzeIntervalPressure(func, useDef);
      if (peakPressure <= this.config.memoryBudget) break;
      if (candidates.length === 0) break;

      candidates.sort((a, b) => b.score - a.score);
      this._rematerialize(func, candidates[0]);
      changed = true;
      iterations++;
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _analyzeIntervalPressure(func, useDef) {
    const topo = useDef.topologicalOrder;
    const intervals = new Map();

    for (const arg of func.args) {
      intervals.set(arg, { start: -1, end: -1 });
    }

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      for (let j = 0; j < op.numResults; j++) {
        intervals.set(op.getResult(j), { start: i, end: i });
      }
    }

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      for (let j = 0; j < op.numOperands; j++) {
        const val = op.getOperand(j);
        const intv = intervals.get(val);
        if (intv && intv.end < i) intv.end = i;
      }
    }

    const events = [];
    for (const [value, intv] of intervals) {
      if (!(value.type instanceof TensorType)) continue;
      const bytes = value.type.sizeInBytes();
      if (bytes === DYNAMIC || bytes <= 0) continue;
      events.push({ idx: intv.start, delta: bytes, value });
      events.push({ idx: intv.end + 1, delta: -bytes, value: null });
    }
    events.sort((a, b) => a.idx - b.idx || a.delta - b.delta);

    let pressure = 0;
    let peakPressure = 0;
    let peakIdx = 0;
    const liveAtPeak = new Set();
    const currentLive = new Set();

    let ei = 0;
    for (let i = -1; i <= topo.length; i++) {
      while (ei < events.length && events[ei].idx <= i) {
        pressure += events[ei].delta;
        if (events[ei].delta > 0 && events[ei].value) currentLive.add(events[ei].value);
        if (events[ei].delta < 0 && events[ei].value) currentLive.delete(events[ei].value);
        ei++;
      }
      if (pressure > peakPressure) {
        peakPressure = pressure;
        peakIdx = i;
        liveAtPeak.clear();
        for (const v of currentLive) liveAtPeak.add(v);
      }
    }

    const candidates = [];
    for (const value of liveAtPeak) {
      if (!this._canRematerialize(value)) continue;
      const defOp = value.definingOp;
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

    return { peakPressure, peakIdx, candidates };
  }

  _canRematerialize(value) {
    if (value.isBlockArgument()) return false;

    const defOp = value.definingOp;
    if (!defOp) return false;
    if (NEVER_REMAT.has(defOp.opName)) return false;
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

  _computeMemorySaved(value) {
    if (!(value.type instanceof TensorType)) return 0;
    const bytes = value.type.sizeInBytes();
    if (bytes === DYNAMIC) return 0;
    return bytes;
  }

  _operandExtensionCost(defOp, cloneIdx, opOrder) {
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
          const bytes = operand.type.sizeInBytes();
          if (bytes !== DYNAMIC) cost += bytes;
        }
      }
    }
    return cost;
  }

  _estimateRecomputeCost(op) {
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
          const n = t.numel();
          if (n !== DYNAMIC) { elements = n; break; }
        }
      }
      return elements || 1;
    }

    if (def.isReduction) {
      for (let i = 0; i < op.numOperands; i++) {
        const t = op.getOperand(i).type;
        if (t instanceof TensorType) {
          const n = t.numel();
          if (n !== DYNAMIC) return n;
        }
      }
    }

    return Infinity;
  }

  _rematerialize(func, candidate) {
    const { value, definingOp } = candidate;
    const uses = [];
    for (const use of value.uses()) {
      uses.push({ user: use.user, operandIndex: use.operandIndex });
    }

    if (uses.length <= 1) return;

    const opOrder = new Map();
    let idx = 0;
    for (const op of func.ops()) {
      opOrder.set(op, idx++);
    }

    uses.sort((a, b) => (opOrder.get(a.user) || 0) - (opOrder.get(b.user) || 0));

    const lastUseIdx = opOrder.get(uses[uses.length - 1].user) || 0;
    const operandsCostIfExtended = this._operandExtensionCost(definingOp, lastUseIdx, opOrder);
    const savings = this._computeMemorySaved(value);
    if (operandsCostIfExtended >= savings) return;

    for (let i = 1; i < uses.length; i++) {
      const use = uses[i];
      const user = use.user;
      if (!user.parentBlock) continue;

      const clonedOp = new Operation(
        definingOp.opName,
        [...definingOp.operands],
        definingOp.results.map(r => r.type),
        new Map(definingOp.attributes)
      );

      user.parentBlock.insertBefore(clonedOp, user);
      user.replaceOperand(use.operandIndex, clonedOp.getResult(value.resultIndex));
    }
  }
}
