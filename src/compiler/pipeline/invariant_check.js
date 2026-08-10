import { verifyIR } from '../ir/verify.js';
import { CompilationError } from './trace.js';

export const VerifyLevel = Object.freeze({
  OFF: 'off',
  BOUNDARIES: 'boundaries',
  EACH_PASS: 'each-pass',
});

const LEVELS = new Set(Object.values(VerifyLevel));

export function normalizeVerifyLevel(value) {
  const level = value ?? VerifyLevel.EACH_PASS;
  if (!LEVELS.has(level)) {
    throw new Error(`Invalid verify level '${value}'; expected one of ${[...LEVELS].join(', ')}`);
  }
  return level;
}

export function checkIRInvariants(irLevel, target, name, passName = null) {
  const found = verifyIR(irLevel, target);
  if (found.length === 0) return null;
  const prefix = passName ? `pass '${passName}' produced invalid IR: ` : '';
  return new CompilationError('verification', name, prefix + found.join('; '), passName);
}
