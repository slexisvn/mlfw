import { TensorType } from '../../ir/graph/types.js';
import { layoutEquals } from '../../analysis/layout_analysis.js';
import { registry } from '../../ir/graph/ops.js';
import { OpAttrKey } from '../../ir/graph/op_traits.js';
import { LayoutPreference } from '../../ir/graph/layout_pref.js';
import type { Layout, IRType } from '../../ir/graph/types.js';
import type { InferLayoutFn } from '../../ir/graph/layout_pref.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { CompileTarget } from '../../support/config_types.js';

export { LayoutPreference } from '../../ir/graph/layout_pref.js';

export type LayoutPolicyTarget = CompileTarget & { cacheLineBytes?: number };
export type LayoutRuleFn = (op: Operation, target: LayoutPolicyTarget) => LayoutPreference | null;

const DEFAULT_CACHE_LINE_BYTES = 64;
const BYTES_PER_F32 = 4;

export class LayoutPolicy {
  target: LayoutPolicyTarget;
  private _overrides: Map<string, LayoutRuleFn>;

  constructor(target: LayoutPolicyTarget) {
    this.target = target;
    this._overrides = new Map();
  }

  registerRule(opName: string, fn: LayoutRuleFn): void {
    this._overrides.set(opName, fn);
  }

  getPreference(op: Operation): LayoutPreference | null {
    const override = this._overrides.get(op.opName);
    if (override) return override(op, this.target);
    const def = registry.get(op.opName);
    const infer = def === null ? null : def.getAttr<InferLayoutFn>(OpAttrKey.INFER_LAYOUT);
    return infer ? infer(op, this.target) : null;
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
    const def = registry.get(consumer.opName);
    const sensitivity = def === null ? null : def.getAttr<number>(OpAttrKey.LAYOUT_SENSITIVITY);
    if (sensitivity !== null) return numEl * sensitivity * useCount;
    const cacheLineBytes = this.target.cacheLineBytes || DEFAULT_CACHE_LINE_BYTES;
    if (numEl * BYTES_PER_F32 <= cacheLineBytes * BYTES_PER_F32) return 0;
    return Math.floor(numEl * 0.5);
  }
}

