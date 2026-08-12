import type { LirFuncPass } from '../passes/lir_pass.js';
import type { CompilerConfig, PassPhase } from './pipeline_types.js';

export type LirPassFactory = (config: CompilerConfig) => LirFuncPass | null | undefined;
export type LirPassEntry = { factory: LirPassFactory; phase: PassPhase; priority: number };

const _lirPasses: LirPassEntry[] = [];

export function registerLirPass(factory: LirPassFactory, { phase = 'post', priority = 0 }: { phase?: PassPhase; priority?: number } = {}): void {
  _lirPasses.push({ factory, phase, priority });
}

export function snapshotLirPasses(): LirPassEntry[] {
  return [..._lirPasses];
}

export function lirPassesForPhase(phase: PassPhase, config: CompilerConfig, entries: readonly LirPassEntry[] = _lirPasses): LirFuncPass[] {
  return entries
    .filter((e) => e.phase === phase)
    .sort((a, b) => a.priority - b.priority)
    .map((e) => e.factory(config))
    .filter((p): p is LirFuncPass => !!p);
}

export function clearLirPasses(): void {
  _lirPasses.length = 0;
}
