import { Layout, TensorType } from '../../ir/graph/types.js';
import type { IRType } from '../../ir/graph/types.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { CompileTarget } from '../../pipeline/pipeline_types.js';

export type LayoutPolicyTarget = CompileTarget & { cacheLineBytes?: number };
export type LayoutRuleFn = (op: Operation, target: LayoutPolicyTarget) => LayoutPreference | null;

export class LayoutPreference {
  inputs: readonly (Layout | null)[];
  outputs: readonly (Layout | null)[];
  cost: number;

  constructor(inputs: readonly (Layout | null)[], outputs: readonly (Layout | null)[], cost = 0) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.cost = cost;
  }
}

export class LayoutPolicy {
  target: LayoutPolicyTarget;
  private _rules: Map<string, LayoutRuleFn>;

  constructor(target: LayoutPolicyTarget) {
    this.target = target;
    this._rules = new Map();
    this._initDefaultRules();
  }

  registerRule(opName: string, fn: LayoutRuleFn): void {
    this._rules.set(opName, fn);
  }

  getPreference(op: Operation): LayoutPreference | null {
    const rule = this._rules.get(op.opName);
    if (rule) return rule(op, this.target);
    return null;
  }

  estimateConversionCost(fromLayout: Layout | null, toLayout: Layout | null, tensorType: IRType): number {
    if (!(tensorType instanceof TensorType)) return 0;
    if (layoutEquals(fromLayout, toLayout)) return 0;
    const numEl = tensorType.numel();
    if (numEl < 0) return 1024;
    return numEl * 2;
  }

  estimateBenefit(consumer: Operation, tensorType: IRType, useCount: number): number {
    if (!(tensorType instanceof TensorType)) return 0;
    const numEl = tensorType.numel();
    if (numEl < 0) return 0;
    const opName = consumer.opName;
    if (opName === 'dot' || opName === 'conv' || opName === 'matmul') return numEl * 4 * useCount;
    if (opName === 'reduce') return numEl * 2 * useCount;
    const cacheLineBytes = this.target.cacheLineBytes || 64;
    if (numEl * 4 <= cacheLineBytes * 4) return 0;
    return Math.floor(numEl * 0.5);
  }

  _initDefaultRules(): void {
    this._rules.set('conv', (op, tgt) => {
      const inp = op.getOperand(0).type as TensorType;
      const rank = inp?.rank || 4;

      if (tgt.preferredConvLayout) {
        const pref = tgt.preferredConvLayout as unknown as Layout;
        return new LayoutPreference([pref, null], [pref]);
      }

      if (tgt.isGPU() && rank === 4) {
        const nhwc = new Layout([0, 2, 3, 1]);
        return new LayoutPreference([nhwc, null], [nhwc]);
      }

      if (tgt.isCPU() && rank === 4) {
        const nhwc = new Layout([0, 2, 3, 1]);
        return new LayoutPreference([nhwc, null], [nhwc]);
      }
      return null;
    });

    this._rules.set('dot', (op, tgt) => {
      const lhsType = op.getOperand(0).type as TensorType;
      const rhsType = op.getOperand(1).type as TensorType;
      if (!lhsType || !rhsType) return null;
      const lhsLayout = Layout.rowMajor(lhsType.rank);
      if (tgt.isCPU() && rhsType.rank === 2) {
        const colMajor = Layout.columnMajor(rhsType.rank);
        return new LayoutPreference([lhsLayout, colMajor], [lhsLayout]);
      }
      const rhsLayout = Layout.rowMajor(rhsType.rank);
      return new LayoutPreference([lhsLayout, rhsLayout], [lhsLayout]);
    });

    this._rules.set('reduce', (op, tgt) => {
      const outType = op.getResult(0).type as TensorType;
      if (!outType) return null;
      return new LayoutPreference([null], [Layout.rowMajor(outType.rank)]);
    });
  }
}

function layoutEquals(a: Layout | null, b: Layout | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a.equals === 'function') return a.equals(b);
  return false;
}
