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
    this.trace = null;
  }

  run(target, analysisManager) {
    throw new Error('Not implemented');
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
