import { UseDefAnalysis } from './use_def.js';
import { Layout, TensorType } from '../ir/graph/types.js';
import { registry } from '../ir/graph/ops.js';
import { OpTrait } from '../ir/graph/op_registry.js';

export class LayoutAnalysisResult {
  constructor(assignments, conversions, totalCost) {
    this.assignments = assignments;
    this.conversions = conversions;
    this.totalCost = totalCost;
  }
}

export class LayoutAnalysis {
  static get name() { return 'layout'; }
  static get depKey() { return 'layout'; }
  static get dependencies() { return [UseDefAnalysis]; }

  static compute(func, deps, policy) {
    const useDef = deps.useDef;
    const assignments = new Map();
    const topo = useDef.topologicalOrder;

    for (const arg of func.args) {
      if (arg.type instanceof TensorType) {
        assignments.set(arg, arg.type.layout || Layout.rowMajor(arg.type.rank));
      }
    }

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      if (op.opName === 'return' || op.opName === 'yield') continue;

      const pref = policy ? policy.getPreference(op) : null;

      if (pref && pref.outputs.length > 0) {
        for (let r = 0; r < op.numResults; r++) {
          const val = op.getResult(r);
          if (!(val.type instanceof TensorType)) continue;
          const prefLayout = pref.outputs[r] || null;
          if (prefLayout) {
            assignments.set(val, prefLayout);
          } else {
            assignments.set(val, resolveFromInputs(op, assignments));
          }
        }
        continue;
      }

      const def = registry.get(op.opName);
      const isEW = def && def.hasTrait(OpTrait.ELEMENTWISE);

      for (let r = 0; r < op.numResults; r++) {
        const val = op.getResult(r);
        if (!(val.type instanceof TensorType)) continue;
        if (isEW) {
          assignments.set(val, resolveFromInputs(op, assignments));
        } else {
          assignments.set(val, Layout.rowMajor(val.type.rank));
        }
      }
    }

    const conversions = [];
    let totalCost = 0;

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      const pref = policy ? policy.getPreference(op) : null;
      if (!pref) continue;

      for (let j = 0; j < op.numOperands; j++) {
        const operand = op.getOperand(j);
        if (!(operand.type instanceof TensorType)) continue;
        const producerLayout = assignments.get(operand);
        if (!producerLayout) continue;

        const expectedLayout = pref.inputs[j];
        if (!expectedLayout) continue;
        if (layoutEquals(producerLayout, expectedLayout)) continue;

        const fromLayout = toPlainLayout(producerLayout);
        const toLayout = toPlainLayout(expectedLayout);
        if (!fromLayout || !toLayout) continue;
        if (fromLayout.equals(toLayout)) continue;

        conversions.push({
          value: operand,
          consumer: op,
          operandIdx: j,
          from: fromLayout,
          to: toLayout
        });
        if (policy) {
          totalCost += policy.estimateConversionCost(fromLayout, toLayout, operand.type);
        }
      }
    }

    return new LayoutAnalysisResult(assignments, conversions, totalCost);
  }
}

function resolveFromInputs(op, assignments) {
  const counts = new Map();
  let best = null;
  let bestCount = 0;

  for (let i = 0; i < op.numOperands; i++) {
    const operand = op.getOperand(i);
    const layout = assignments.get(operand);
    if (!layout) continue;
    const key = layout.hash ? layout.hash() : 0;
    const count = (counts.get(key) || 0) + 1;
    counts.set(key, count);
    if (count > bestCount) {
      bestCount = count;
      best = layout;
    }
  }

  if (best) return best;
  const firstResult = op.getResult(0);
  if (firstResult && firstResult.type instanceof TensorType) {
    return Layout.rowMajor(firstResult.type.rank);
  }
  return Layout.rowMajor(1);
}

function layoutEquals(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a.equals === 'function') return a.equals(b);
  return false;
}

function toPlainLayout(layout) {
  if (layout instanceof Layout) return layout;
  if (layout && typeof layout.toLayout === 'function') {
    try { return layout.toLayout(); } catch { /* blocked layout */ }
  }
  if (layout && layout.baseOrder && !layout.isBlocked?.()) {
    return new Layout(layout.baseOrder);
  }
  if (layout && layout.order) return new Layout(layout.order);
  return null;
}
