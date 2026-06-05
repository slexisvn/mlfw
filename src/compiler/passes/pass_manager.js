import { performance } from 'node:perf_hooks';
import { AnalysisManager } from '../analysis/analysis_manager.js';
import { FunctionPass, ModulePass, PassResult } from './pass.js';
import { TraceLevel } from '../pipeline/trace.js';
import { CompilationError } from '../pipeline/trace.js';

function countOps(target) {
  if (typeof target.numOps === 'function') return target.numOps();
  if (typeof target[Symbol.iterator] === 'function') {
    let total = 0;
    for (const func of target) {
      if (typeof func.numOps === 'function') total += func.numOps();
    }
    return total;
  }
  return -1;
}

export class PassManager {
  constructor() {
    this.passes = [];
    this.analysisManager = new AnalysisManager();
    this.trace = null;
  }

  addPass(pass) {
    this.passes.push(pass);
  }

  setTrace(trace) {
    this.trace = trace;
  }

  run(module, options = {}) {
    let anyChanged = false;
    const results = [];
    const verbose = this.trace && this.trace.level >= TraceLevel.VERBOSE;
    const resilient = options.errorMode === 'resilient';
    const errors = [];
    const failedFunctions = new Set();

    for (const pass of this.passes) {
      if (this.trace) pass.trace = this.trace;

      if (pass instanceof ModulePass) {
        const opsBefore = verbose ? countOps(module) : -1;
        const t0 = verbose ? performance.now() : 0;

        const result = pass.run(module, this.analysisManager);
        results.push(result);

        if (verbose) {
          this.trace.passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(module));
        }

        if (result === PassResult.CHANGED) {
          anyChanged = true;
          this.analysisManager.invalidateAll();
        }
      } else if (pass instanceof FunctionPass) {
        let passChanged = false;

        for (const func of module) {
          if (failedFunctions.has(func.name)) continue;

          const opsBefore = verbose ? countOps(func) : -1;
          const t0 = verbose ? performance.now() : 0;

          if (resilient) {
            try {
              const result = pass.run(func, this.analysisManager);
              if (verbose) this.trace.passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
              if (result === PassResult.CHANGED) {
                passChanged = true;
                anyChanged = true;
                this.analysisManager.invalidate(func, pass.preservedAnalyses);
                func.bumpVersion();
              }
            } catch (e) {
              errors.push(new CompilationError('graphPasses', func.name, e.message, pass.name));
              failedFunctions.add(func.name);
            }
          } else {
            const result = pass.run(func, this.analysisManager);
            if (verbose) this.trace.passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
            if (result === PassResult.CHANGED) {
              passChanged = true;
              anyChanged = true;
              this.analysisManager.invalidate(func, pass.preservedAnalyses);
              func.bumpVersion();
            }
          }
        }
        results.push(passChanged ? PassResult.CHANGED : PassResult.UNCHANGED);
      }

      pass.trace = null;
    }

    return { changed: anyChanged, results, errors: errors.length > 0 ? errors : null, failedFunctions: failedFunctions.size > 0 ? failedFunctions : null };
  }
}
