function toMap(value) {
  if (value instanceof Map) return value;
  if (value && typeof value === 'object') return new Map(Object.entries(value));
  return new Map();
}

export class CompilerContext {
  constructor({ loweringRules = null, codegenEntries = null } = {}) {
    this.loweringRules = toMap(loweringRules);
    this.codegenEntries = toMap(codegenEntries);
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
}
