export const PassResult = Object.freeze({
  UNCHANGED: 0,
  CHANGED: 1,
  FAILED: 2
});

export class Pass {
  constructor(name) {
    this.name = name;
    this.preservedAnalyses = new Set();
    this.invalidatedAnalyses = new Set();
    this.requiredAnalyses = [];
    this.optLevel = 0;
    this.trace = null;
  }

  run(target, analysisManager) {
    throw new Error('Not implemented');
  }
}

export class PassContext {
  constructor({ optLevel = Infinity, disabledPasses = [], config = {} } = {}) {
    this.optLevel = optLevel;
    this.disabledPasses = disabledPasses instanceof Set ? disabledPasses : new Set(disabledPasses);
    this.config = config instanceof Map ? config : new Map(Object.entries(config));
  }

  shouldRun(pass) {
    if (this.disabledPasses.has(pass.name)) return false;
    if ((pass.optLevel || 0) > this.optLevel) return false;
    return true;
  }
}

export class FunctionPass extends Pass {
  run(func, analysisManager) {
    throw new Error('Not implemented');
  }
}

export class ModulePass extends Pass {
  run(module, analysisManager) {
    throw new Error('Not implemented');
  }
}
