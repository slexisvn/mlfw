import { AnalysisManager } from '../analysis/analysis_manager.js';
import { FunctionPass, ModulePass, PassResult } from './pass.js';

export class PassManager {
  constructor() {
    this.passes = [];
    this.analysisManager = new AnalysisManager();
    this.instrumentation = null;
  }

  addPass(pass) {
    this.passes.push(pass);
  }

  setInstrumentation(inst) {
    this.instrumentation = inst;
  }

  run(module, options = {}) {
    let anyChanged = false;
    const results = [];

    for (const pass of this.passes) {
      if (pass instanceof ModulePass) {
        if (this.instrumentation && this.instrumentation.beforePass) {
          this.instrumentation.beforePass(pass, module);
        }
        
        const result = pass.run(module, this.analysisManager);
        results.push(result);
        
        if (result === PassResult.CHANGED) {
          anyChanged = true;
          this.analysisManager.invalidateAll();
        }
        
        if (this.instrumentation && this.instrumentation.afterPass) {
          this.instrumentation.afterPass(pass, module, result);
        }
      } else if (pass instanceof FunctionPass) {
        let passChanged = false;
        
        for (const func of module) {
          if (this.instrumentation && this.instrumentation.beforePass) {
            this.instrumentation.beforePass(pass, func);
          }
          
          const result = pass.run(func, this.analysisManager);
          
          if (result === PassResult.CHANGED) {
            passChanged = true;
            anyChanged = true;
            this.analysisManager.invalidate(func, pass.preservedAnalyses);
            func.bumpVersion();
          }
          
          if (this.instrumentation && this.instrumentation.afterPass) {
            this.instrumentation.afterPass(pass, func, result);
          }
        }
        results.push(passChanged ? PassResult.CHANGED : PassResult.UNCHANGED);
      }
    }

    return { changed: anyChanged, results };
  }
}
