import type { CompilerConfig, PassPhase, TirPass } from './pipeline_types.js';

export type TirPassFactory = (config: CompilerConfig) => TirPass | null | undefined;
export type TirPassEntry = { factory: TirPassFactory; phase: PassPhase; priority: number };

const _tirPasses: TirPassEntry[] = [];

export function registerTirPass(factory: TirPassFactory, { phase = 'post', priority = 0 }: { phase?: PassPhase; priority?: number } = {}): void {
  _tirPasses.push({ factory, phase, priority });
}

export function snapshotTirPasses(): TirPassEntry[] {
  return [..._tirPasses];
}

export function tirPassesForPhase(phase: PassPhase, config: CompilerConfig, entries: readonly TirPassEntry[] = _tirPasses): TirPass[] {
  return entries
    .filter((e) => e.phase === phase)
    .sort((a, b) => a.priority - b.priority)
    .map((e) => e.factory(config))
    .filter((p): p is TirPass => !!p);
}

export function clearTirPasses(): void {
  _tirPasses.length = 0;
}
