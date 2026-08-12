import { snapshotGraphPasses, graphPassesForPhase } from './graph_pass_registry.js';
import type { GraphPassEntry } from './graph_pass_registry.js';
import type { CompilerConfig, CompileTarget, GraphPass, PassPhase } from './pipeline_types.js';
import type { LoweringRuleFn } from '../passes/lowering/lowering_registry.js';

export type CompilerContextOpts = {
  loweringRules?: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  codegenEntries?: ReadonlyMap<string, unknown> | Record<string, unknown> | null;
  graphPasses?: GraphPassEntry[] | null;
};

function toMap(value: ReadonlyMap<string, unknown> | Record<string, unknown> | null | undefined): Map<string, unknown> {
  if (value instanceof Map) return value as Map<string, unknown>;
  if (value && typeof value === 'object') return new Map(Object.entries(value));
  return new Map();
}

export class CompilerContext {
  loweringRules: Map<string, unknown>;
  codegenEntries: Map<string, unknown>;
  graphPasses: GraphPassEntry[];

  constructor({ loweringRules = null, codegenEntries = null, graphPasses = null }: CompilerContextOpts = {}) {
    this.loweringRules = toMap(loweringRules);
    this.codegenEntries = toMap(codegenEntries);
    this.graphPasses = graphPasses || snapshotGraphPasses();
  }

  get hasOverrides(): boolean {
    return this.loweringRules.size > 0 || this.codegenEntries.size > 0;
  }

  getLoweringRule(opName: string): LoweringRuleFn | null {
    return (this.loweringRules.get(opName) as LoweringRuleFn) || null;
  }

  getCodegenEntry(targetKind: string): unknown {
    return this.codegenEntries.get(targetKind) || null;
  }

  passesForPhase(phase: PassPhase, config: CompilerConfig, target: CompileTarget): GraphPass[] {
    return graphPassesForPhase(phase, config, target, this.graphPasses);
  }
}
