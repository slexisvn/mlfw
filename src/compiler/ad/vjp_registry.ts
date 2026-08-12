import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { IRBuilder } from '../ir/graph/builder.js';
import type { AttrValue, TensorType } from '../ir/graph/types.js';

export type TensorValue = Value & { readonly type: TensorType };

export type VJPContext = {
  builder: IRBuilder;
  op: Operation;
  operands: TensorValue[];
  results: TensorValue[];
  gradOutputs: (TensorValue | null)[];
  attrs: ReadonlyMap<string, AttrValue>;
  full: (value: number, type: TensorType) => TensorValue;
};

export type VJPRule = (ctx: VJPContext) => (Value | null)[] | null | undefined;
export type RegionVJPRule = (ctx: VJPContext) => (Value | null)[] | null | undefined;

const _rules = new Map<string, VJPRule>();
const _barriers = new Set<string>();

export function registerVJPRule(opName: string, ruleFn: VJPRule): void {
  _rules.set(opName, ruleFn);
}

export function getVJPRule(opName: string): VJPRule | null {
  return _rules.get(opName) || null;
}

export function requireVJPRuleOrBarrier(opName: string): VJPRule | null {
  const rule = _rules.get(opName);
  if (rule) return rule;
  if (_barriers.has(opName)) return null;
  throw new Error(`autodiff: op '${opName}' is on the gradient path but has no VJP rule and is not a registered gradient barrier. Register one with registerVJPRule('${opName}', ...) or registerGradientBarrier('${opName}').`);
}

export function findUnsupportedGradOps(ops: Iterable<Operation>): string[] {
  const missing = new Set<string>();
  for (const op of ops) {
    if (!_rules.has(op.opName) && !_barriers.has(op.opName)) missing.add(op.opName);
  }
  return [...missing];
}

export function hasVJPRule(opName: string): boolean {
  return _rules.has(opName);
}

export function listRegisteredOps(): string[] {
  return [..._rules.keys()];
}

const _regionRules = new Map<string, RegionVJPRule>();

export function registerRegionVJP(opName: string, fn: RegionVJPRule): void {
  _regionRules.set(opName, fn);
}

export function getRegionVJP(opName: string): RegionVJPRule | null {
  return _regionRules.get(opName) || null;
}

export function registerGradientBarrier(opName: string): void {
  _barriers.add(opName);
}

export function isGradientBarrier(opName: string): boolean {
  return _barriers.has(opName);
}
