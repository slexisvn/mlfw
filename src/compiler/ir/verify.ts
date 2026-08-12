import { verifyModule, verifyFunction } from './graph/verifier.js';
import { TensorVerifier } from './tensor/verifier.js';
import { verifyLIR } from './lir/verifier.js';
import type { GraphModule } from './graph/module.js';
import type { GraphFunction } from './graph/function.js';
import type { PrimFunc } from './tensor/nodes.js';
import type { LIRFunc } from './lir/nodes.js';

export const IRLevel = Object.freeze({
  GRAPH_MODULE: 'graph-module',
  GRAPH_FUNC: 'graph-func',
  TIR: 'tir',
  LIR: 'lir',
});

export type IRLevelValue = (typeof IRLevel)[keyof typeof IRLevel];
export type IRVerifyFn = (target: never) => string[];

const _verifiers = new Map<string, IRVerifyFn>();

export function registerIRVerifier<T>(level: IRLevelValue, verify: (target: T) => string[]): (target: T) => string[] {
  _verifiers.set(level, verify as IRVerifyFn);
  return verify;
}

export function unregisterIRVerifier(level: IRLevelValue): boolean {
  return _verifiers.delete(level);
}

export function getIRVerifier(level: IRLevelValue): IRVerifyFn | null {
  return _verifiers.get(level) || null;
}

export function verifyIR(level: IRLevelValue, target: unknown): string[] {
  const verify = _verifiers.get(level);
  if (!verify) throw new Error(`No IR verifier registered for level '${level}'`);
  return (verify as (t: unknown) => string[])(target);
}

export function irLevels(): string[] {
  return [..._verifiers.keys()];
}

function toMessages(errors: readonly unknown[]): string[] {
  const out: string[] = [];
  for (const e of errors) out.push(typeof e === 'string' ? e : String(e));
  return out;
}

registerIRVerifier<GraphModule>(IRLevel.GRAPH_MODULE, (module) => toMessages(verifyModule(module)));
registerIRVerifier<GraphFunction>(IRLevel.GRAPH_FUNC, (func) => toMessages(verifyFunction(func)));
registerIRVerifier<PrimFunc>(IRLevel.TIR, (primFunc) => toMessages(new TensorVerifier().verify(primFunc)));
registerIRVerifier<LIRFunc>(IRLevel.LIR, (lirFunc) => toMessages(verifyLIR(lirFunc)));
