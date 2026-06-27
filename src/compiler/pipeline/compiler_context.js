import { snapshotGraphPasses, graphPassesForPhase } from './graph_pass_registry.js';

function toMap(value) {
  if (value instanceof Map) return value;
  if (value && typeof value === 'object') return new Map(Object.entries(value));
  return new Map();
}

export class CompilerContext {
  constructor({ loweringRules = null, codegenEntries = null, graphPasses = null } = {}) {
    this.loweringRules = toMap(loweringRules);
    this.codegenEntries = toMap(codegenEntries);
    this.graphPasses = graphPasses || snapshotGraphPasses();
  }

  get hasOverrides() {
    return this.loweringRules.size > 0 || this.codegenEntries.size > 0;
  }

  getLoweringRule(opName) {
    return this.loweringRules.get(opName) || null;
  }

  getCodegenEntry(targetKind) {
    return this.codegenEntries.get(targetKind) || null;
  }

  passesForPhase(phase, config, target) {
    return graphPassesForPhase(phase, config, target, this.graphPasses);
  }
}
