import { verifyIR } from 'mlfw/compiler/ir/verify.js';
import type { IRLevelName, VerifyReport } from '../protocol.js';

const GRAPH_LEVELS: ReadonlySet<string> = new Set(['graph-module', 'graph-func']);

export function invariantsOf(level: IRLevelName, target: unknown): string[] {
  if (!target) return [];
  try {
    if (GRAPH_LEVELS.has(level)) return verifyIR(level, target);

    const found: string[] = [];
    for (const func of target as Iterable<{ name: string }>) {
      for (const message of verifyIR(level, func)) found.push(`${func.name}: ${message}`);
    }
    return found;
  } catch (error) {
    return [`the verifier itself threw: ${error instanceof Error ? error.message : String(error)}`];
  }
}

export function verifyReport(before: readonly string[], after: readonly string[]): VerifyReport {
  const already = new Set(before);
  const introduced: string[] = [];
  const carried: string[] = [];

  for (const message of after) {
    if (already.has(message)) carried.push(message);
    else introduced.push(message);
  }

  return { introduced, carried };
}
