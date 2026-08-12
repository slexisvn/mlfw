import type { CompilerConfig, CompileTarget, GraphPass, PassPhase } from './pipeline_types.js';

export type GraphPassFactory = (config: CompilerConfig, target: CompileTarget) => GraphPass | null | undefined;
export type GraphPassEntry = { factory: GraphPassFactory; phase: PassPhase; priority: number };

const _graphPasses: GraphPassEntry[] = [];

export function registerGraphPass(factory: GraphPassFactory, { phase = 'post', priority = 0 }: { phase?: PassPhase; priority?: number } = {}): void {
  _graphPasses.push({ factory, phase, priority });
}

export function snapshotGraphPasses(): GraphPassEntry[] {
  return [..._graphPasses];
}

export function graphPassesForPhase(phase: PassPhase, config: CompilerConfig, target: CompileTarget, entries: readonly GraphPassEntry[] = _graphPasses): GraphPass[] {
  return entries
    .filter((e) => e.phase === phase)
    .sort((a, b) => a.priority - b.priority)
    .map((e) => e.factory(config, target))
    .filter((p): p is GraphPass => !!p);
}

export function clearGraphPasses(): void {
  _graphPasses.length = 0;
}
