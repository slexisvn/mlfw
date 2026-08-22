import { UseDefAnalysis } from './use_def.js';
import { Layout, TensorType } from '../ir/graph/types.js';
import { registry } from '../ir/graph/ops.js';
import { OpTrait } from '../ir/graph/op_registry.js';
import { isTerminatorOp } from '../ir/graph/op_traits.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { AnalysisCtor, AnalysisDeps } from './analysis_manager.js';
import type { UseDefResult } from './use_def.js';

export type LayoutLike = {
  equals?: (other: unknown) => boolean;
  hash?: () => number;
  toLayout?: () => Layout;
  baseOrder?: readonly number[];
  order?: readonly number[];
  isBlocked?: () => boolean;
};

export type LayoutPreference = { inputs: readonly (LayoutLike | null)[]; outputs: readonly (LayoutLike | null)[] };

export type LayoutPolicy = {
  getPreference(op: Operation): LayoutPreference | null;
  estimateConversionCost(from: Layout, to: Layout, type: TensorType): number;
};

export type LayoutConversion = {
  value: Value;
  consumer: Operation;
  operandIdx: number;
  from: Layout;
  to: Layout;
};

export class LayoutAnalysisResult {
  assignments: Map<Value, LayoutLike>;
  conversions: LayoutConversion[];
  totalCost: number;

  constructor(assignments: Map<Value, LayoutLike>, conversions: LayoutConversion[], totalCost: number) {
    this.assignments = assignments;
    this.conversions = conversions;
    this.totalCost = totalCost;
  }
}

export class LayoutAnalysis {
  static get name(): string { return 'layout'; }
  static get depKey(): string { return 'layout'; }
  static get dependencies(): readonly AnalysisCtor[] { return [UseDefAnalysis as unknown as AnalysisCtor]; }

  static compute(func: GraphFunction, deps: AnalysisDeps, policy?: LayoutPolicy | null): LayoutAnalysisResult {
    const useDef = deps.useDef as UseDefResult;
    const assignments = new Map<Value, LayoutLike>();
    const topo = useDef.topologicalOrder;

    for (const arg of func.args) {
      if (arg.type instanceof TensorType) {
        assignments.set(arg, (arg.type as TensorType).layout || Layout.rowMajor((arg.type as TensorType).rank));
      }
    }

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      if (isTerminatorOp(op.opName)) continue;

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
          assignments.set(val, Layout.rowMajor((val.type as TensorType).rank));
        }
      }
    }

    const conversions: LayoutConversion[] = [];
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
          totalCost += policy.estimateConversionCost(fromLayout, toLayout, operand.type as TensorType);
        }
      }
    }

    return new LayoutAnalysisResult(assignments, conversions, totalCost);
  }
}

function resolveFromInputs(op: Operation, assignments: ReadonlyMap<Value, LayoutLike>): LayoutLike {
  const counts = new Map<number, number>();
  let best: LayoutLike | null = null;
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
    return Layout.rowMajor((firstResult.type as TensorType).rank);
  }
  return Layout.rowMajor(1);
}

export function layoutEquals(a: LayoutLike | null | undefined, b: LayoutLike | null | undefined): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a.equals === 'function') return a.equals(b);
  return false;
}

function toPlainLayout(layout: LayoutLike | null | undefined): Layout | null {
  if (layout instanceof Layout) return layout;
  if (layout && typeof layout.toLayout === 'function') {
    try { return layout.toLayout(); } catch { /* blocked layout */ }
  }
  if (layout && layout.baseOrder && !layout.isBlocked?.()) {
    return new Layout(layout.baseOrder as readonly number[]);
  }
  if (layout && layout.order) return new Layout(layout.order);
  return null;
}
