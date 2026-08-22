import { checkIRInvariants } from '../pipeline/invariant_check.js';
import type { CompilationError } from '../pipeline/trace.js';
import type { IRLevelValue } from '../ir/verify.js';
import type { TraceLog } from '../pipeline/trace.js';

export type NamedFunc = { name: string };
export type VerifyCtx = {
  trace: TraceLog;
  errors: CompilationError[];
  failed: Set<string>;
  resilient: boolean;
};

export abstract class PassManagerBase<P> {
  passes: P[];
  trace: TraceLog | null;
  checkEachPass: boolean;

  constructor() {
    this.passes = [];
    this.trace = null;
    this.checkEachPass = false;
  }

  addPass(pass: P): void {
    this.passes.push(pass);
  }

  setTrace(trace: TraceLog | null): void {
    this.trace = trace;
  }

  setCheckEachPass(enabled: boolean): void {
    this.checkEachPass = enabled;
  }

  protected _verifyFuncs(level: IRLevelValue, funcs: Iterable<NamedFunc>, ctx: VerifyCtx, passName: string): void {
    for (const func of funcs) {
      if (ctx.failed.has(func.name)) continue;
      const err = checkIRInvariants(level, func, func.name, passName);
      if (!err) continue;
      ctx.errors.push(err);
      ctx.failed.add(func.name);
      ctx.trace.errorEvent('verification', func.name, err.message);
      if (!ctx.resilient) throw new Error(err.toString());
    }
  }
}
