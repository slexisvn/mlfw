import { registry } from './ops.js';

export function isConstantOp(opName) {
  const def = registry.get(opName);
  return def !== null && def.isConstant;
}

export function isTerminatorOp(opName) {
  const def = registry.get(opName);
  return def !== null && def.isTerminator;
}

export function isBroadcastOp(opName) {
  const def = registry.get(opName);
  return def !== null && def.isBroadcast;
}

export function isReductionOp(opName) {
  const def = registry.get(opName);
  return def !== null && def.isReduction;
}
