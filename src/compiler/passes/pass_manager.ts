import { AnalysisManager } from '../analysis/analysis_manager.js';
import { PassManagerBase } from './pass_manager_base.js';
import { FunctionPass, ModulePass, PassResult, PassContext } from './pass.js';
import { TraceLevel } from '../pipeline/trace.js';
import { CompilationError } from '../pipeline/trace.js';
import { checkIRInvariants } from '../pipeline/invariant_check.js';
import { IRLevel } from '../ir/verify.js';
import { cloneGraphModule } from '../ir/graph/module.js';
import type { GraphModule } from '../ir/graph/module.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Pass, PassResultValue, PassTarget } from './pass.js';
import type { TraceLog } from '../pipeline/trace.js';

export type PassManagerEntry = Pass | FixedPointGroup;

export type PassRunCtx = {
  verbose: boolean | null;
  resilient: boolean;
  errors: CompilationError[];
  failedFunctions: Set<string>;
  anyChanged: boolean;
  passContext: PassContext;
};

export type PassRunOptions = Readonly<{ errorMode?: string; passContext?: PassContext | null }>;
export type PassRunResult = {
  changed: boolean;
  results: PassResultValue[];
  errors: CompilationError[] | null;
  failedFunctions: Set<string> | null;
};

function countOps(target: PassTarget): number {
  const fn = target as GraphFunction;
  if (typeof fn.numOps === 'function') return fn.numOps();
  const mod = target as GraphModule;
  if (typeof mod[Symbol.iterator] === 'function') {
    let total = 0;
    for (const func of mod) {
      if (typeof func.numOps === 'function') total += func.numOps();
    }
    return total;
  }
  return -1;
}

export class FixedPointGroup {
  name: string;
  passes: Pass[];
  maxIterations: number;

  constructor(name: string, passes: Pass[], maxIterations = 8) {
    this.name = name;
    this.passes = passes;
    this.maxIterations = maxIterations;
  }
}

export class PassManager extends PassManagerBase<PassManagerEntry> {
  analysisManager: AnalysisManager;

  protected override readonly irLevel = IRLevel.GRAPH_MODULE;

  constructor() {
    super();
    this.analysisManager = new AnalysisManager();
  }

  _verifyAfter(pass: Pass, target: PassTarget, isModule: boolean): CompilationError | null {
    if (!this.checkEachPass) return null;
    const level = isModule ? IRLevel.GRAPH_MODULE : IRLevel.GRAPH_FUNC;
    const name = isModule ? (target.name || '<module>') : target.name;
    return checkIRInvariants(level, target, name, pass.name);
  }

  _applyPass(pass: Pass, module: GraphModule, ctx: PassRunCtx, results: PassResultValue[]): { changed: boolean; fatal: boolean } {
    if (this.trace) pass.trace = this.trace;
    this._notifyBefore(pass, module);
    const verbose = ctx.verbose;
    const resilient = ctx.resilient;
    let changed = false;
    let fatal = false;

    if (pass instanceof ModulePass) {
      const opsBefore = verbose ? countOps(module) : -1;
      const t0 = verbose ? performance.now() : 0;

      const snapshot = resilient ? cloneGraphModule(module) : null;

      let result;
      try {
        result = pass.run(module, this.analysisManager);
      } catch (e) {
        if (!resilient) throw e;
        module.restoreFrom(snapshot as GraphModule);
        this.analysisManager.invalidateAll();
        results.push(PassResult.FAILED);
        ctx.errors.push(new CompilationError('graphPasses', module.name || '<module>', (e as Error).message, pass.name));
        this._notifyAfter(pass, module, PassResult.FAILED);
        pass.trace = null;
        return { changed, fatal: false };
      }
      results.push(result);

      if (verbose) (this.trace as TraceLog).passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(module));

      if (result === PassResult.CHANGED) {
        changed = true;
        ctx.anyChanged = true;
        this.analysisManager.invalidateFunctions(module, pass.preservedAnalyses);
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
            for (const A of pass.requiredAnalyses) this.analysisManager.getAnalysis(A, func);
            const result = pass.run(func, this.analysisManager);
            if (verbose) (this.trace as TraceLog).passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
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
            ctx.errors.push(new CompilationError('graphPasses', func.name, (e as Error).message, pass.name));
            ctx.failedFunctions.add(func.name);
          }
        } else {
          for (const A of pass.requiredAnalyses) this.analysisManager.getAnalysis(A, func);
          const result = pass.run(func, this.analysisManager);
          if (verbose) (this.trace as TraceLog).passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
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

    this._notifyAfter(pass, module, changed ? PassResult.CHANGED : PassResult.UNCHANGED);
    pass.trace = null;
    return { changed, fatal };
  }

  _runGroup(group: FixedPointGroup, module: GraphModule, ctx: PassRunCtx, results: PassResultValue[]): boolean {
    const maxIter = group.maxIterations > 0 ? group.maxIterations : 1;
    for (let iter = 0; iter < maxIter; iter++) {
      let iterChanged = false;
      for (const pass of group.passes) {
        if (!ctx.passContext.shouldRun(pass)) continue;
        const { changed, fatal } = this._applyPass(pass, module, ctx, results);
        if (fatal) return true;
        if (changed) iterChanged = true;
      }
      if (!iterChanged) return false;
    }
    if (this.trace) this.trace.passRun(`${group.name}:max-iter`, PassResult.UNCHANGED, 0, -1, -1);
    return false;
  }

  run(module: GraphModule, options: PassRunOptions = {}): PassRunResult {
    const ctx: PassRunCtx = {
      verbose: this.trace && this.trace.level >= TraceLevel.VERBOSE,
      resilient: options.errorMode === 'resilient',
      errors: [],
      failedFunctions: new Set<string>(),
      anyChanged: false,
      passContext: options.passContext || new PassContext(),
    };
    const results: PassResultValue[] = [];

    for (const item of this.passes) {
      if (!(item instanceof FixedPointGroup) && !ctx.passContext.shouldRun(item)) continue;
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
