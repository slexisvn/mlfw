import { registry } from './ops.js';

export function isConstantOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isConstant;
}

export function isTerminatorOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isTerminator;
}

export function isBroadcastOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isBroadcast;
}

export function isReductionOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isReduction;
}
