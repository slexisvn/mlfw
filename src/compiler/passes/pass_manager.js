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

export class FixedPointGroup {
  constructor(name, passes, maxIterations = 8) {
    this.name = name;
    this.passes = passes;
    this.maxIterations = maxIterations;
  }
}

export class PassManager {
  constructor() {
    this.passes = [];
    this.analysisManager = new AnalysisManager();
    this.trace = null;
    this.verifyHook = null;
  }

  addPass(pass) {
    this.passes.push(pass);
  }

  setTrace(trace) {
    this.trace = trace;
  }

  setVerifyHook(hook) {
    this.verifyHook = hook;
  }

  _verifyAfter(pass, target, isModule) {
    if (!this.verifyHook) return null;
    const found = this.verifyHook(target, isModule);
    if (!found || found.length === 0) return null;
    const name = isModule ? (target.name || '<module>') : target.name;
    return new CompilationError('verification', name, `pass '${pass.name}' produced invalid IR: ${found.join('; ')}`, pass.name);
  }

  _applyPass(pass, module, ctx, results) {
    if (this.trace) pass.trace = this.trace;
    const verbose = ctx.verbose;
    const resilient = ctx.resilient;
    let changed = false;
    let fatal = false;

    if (pass instanceof ModulePass) {
      const opsBefore = verbose ? countOps(module) : -1;
      const t0 = verbose ? performance.now() : 0;

      let result;
      try {
        result = pass.run(module, this.analysisManager);
      } catch (e) {
        if (!resilient) throw e;
        this.analysisManager.invalidateAll();
        results.push(PassResult.FAILED);
        ctx.errors.push(new CompilationError('graphPasses', module.name || '<module>', e.message, pass.name));
        return { changed, fatal: false };
      }
      results.push(result);

      if (verbose) this.trace.passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(module));

      if (result === PassResult.CHANGED) {
        changed = true;
        ctx.anyChanged = true;
        this.analysisManager.invalidateAll();
        const verr = this._verifyAfter(pass, module, true);
        if (verr) {
          ctx.errors.push(verr);
          if (!resilient) fatal = true;
        }
      } else if (result === PassResult.FAILED) {
        this.analysisManager.invalidateAll();
        ctx.errors.push(new CompilationError('graphPasses', module.name || '<module>', `pass '${pass.name}' failed`, pass.name));
        if (!resilient) fatal = true;
      }
    } else if (pass instanceof FunctionPass) {
      let passChanged = false;

      for (const func of module) {
        if (ctx.failedFunctions.has(func.name)) continue;

        const opsBefore = verbose ? countOps(func) : -1;
        const t0 = verbose ? performance.now() : 0;

        if (resilient) {
          try {
            const result = pass.run(func, this.analysisManager);
            if (verbose) this.trace.passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
            if (result === PassResult.CHANGED) {
              passChanged = true;
              ctx.anyChanged = true;
              func.bumpVersion();
              this.analysisManager.invalidate(func, pass.preservedAnalyses);
              const verr = this._verifyAfter(pass, func, false);
              if (verr) { ctx.errors.push(verr); ctx.failedFunctions.add(func.name); }
            } else if (result === PassResult.FAILED) {
              this.analysisManager.invalidate(func);
              ctx.errors.push(new CompilationError('graphPasses', func.name, `pass '${pass.name}' failed`, pass.name));
              ctx.failedFunctions.add(func.name);
            }
          } catch (e) {
            ctx.errors.push(new CompilationError('graphPasses', func.name, e.message, pass.name));
            ctx.failedFunctions.add(func.name);
          }
        } else {
          const result = pass.run(func, this.analysisManager);
          if (verbose) this.trace.passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
          if (result === PassResult.CHANGED) {
            passChanged = true;
            ctx.anyChanged = true;
            func.bumpVersion();
            this.analysisManager.invalidate(func, pass.preservedAnalyses);
            const verr = this._verifyAfter(pass, func, false);
            if (verr) {
              ctx.errors.push(verr);
              ctx.failedFunctions.add(func.name);
              fatal = true;
              break;
            }
          } else if (result === PassResult.FAILED) {
            this.analysisManager.invalidate(func);
            ctx.errors.push(new CompilationError('graphPasses', func.name, `pass '${pass.name}' failed`, pass.name));
            ctx.failedFunctions.add(func.name);
            fatal = true;
            break;
          }
        }
      }
      results.push(passChanged ? PassResult.CHANGED : PassResult.UNCHANGED);
      changed = passChanged;
    }

    pass.trace = null;
    return { changed, fatal };
  }

  _runGroup(group, module, ctx, results) {
    const maxIter = group.maxIterations > 0 ? group.maxIterations : 1;
    for (let iter = 0; iter < maxIter; iter++) {
      let iterChanged = false;
      for (const pass of group.passes) {
        const { changed, fatal } = this._applyPass(pass, module, ctx, results);
        if (fatal) return true;
        if (changed) iterChanged = true;
      }
      if (!iterChanged) return false;
    }
    if (this.trace) this.trace.passRun(`${group.name}:max-iter`, PassResult.UNCHANGED, 0, -1, -1);
    return false;
  }

  run(module, options = {}) {
    const ctx = {
      verbose: this.trace && this.trace.level >= TraceLevel.VERBOSE,
      resilient: options.errorMode === 'resilient',
      errors: [],
      failedFunctions: new Set(),
      anyChanged: false,
    };
    const results = [];

    for (const item of this.passes) {
      const fatal = item instanceof FixedPointGroup
        ? this._runGroup(item, module, ctx, results)
        : this._applyPass(item, module, ctx, results).fatal;
      if (fatal) {
        return { changed: ctx.anyChanged, results, errors: ctx.errors, failedFunctions: ctx.failedFunctions.size > 0 ? ctx.failedFunctions : null };
      }
    }

    return {
      changed: ctx.anyChanged,
      results,
      errors: ctx.errors.length > 0 ? ctx.errors : null,
      failedFunctions: ctx.failedFunctions.size > 0 ? ctx.failedFunctions : null,
    };
  }
}
