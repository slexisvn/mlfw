import { verifyIR } from '../ir/verify.js';
import { CompilationError } from './trace.js';
import type { IRLevelValue } from '../ir/verify.js';

export type VerifyLevelValue = (typeof VerifyLevel)[keyof typeof VerifyLevel];

export const VerifyLevel = Object.freeze({
  OFF: 'off',
  BOUNDARIES: 'boundaries',
  EACH_PASS: 'each-pass',
});

const LEVELS = new Set<string>(Object.values(VerifyLevel));

export function normalizeVerifyLevel(value: string | null | undefined): VerifyLevelValue {
  const level = value ?? VerifyLevel.EACH_PASS;
  if (!LEVELS.has(level)) {
    throw new Error(`Invalid verify level '${value}'; expected one of ${[...LEVELS].join(', ')}`);
  }
  return level as VerifyLevelValue;
}

export function checkIRInvariants(irLevel: IRLevelValue, target: unknown, name: string, passName: string | null = null): CompilationError | null {
  const found = verifyIR(irLevel, target);
  if (found.length === 0) return null;
  const prefix = passName ? `pass '${passName}' produced invalid IR: ` : '';
  return new CompilationError('verification', name, prefix + found.join('; '), passName);
}
