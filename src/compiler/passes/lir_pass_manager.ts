import { CompilationError } from '../pipeline/trace.js';
import { checkIRInvariants } from '../pipeline/invariant_check.js';
import { IRLevel } from '../ir/verify.js';
import type { LIRFunc } from '../ir/lir/nodes.js';
import type { LirFuncPass, LirPassCtx } from './lir_pass.js';
import type { TraceLog } from '../pipeline/trace.js';

export type LirRunCtx = LirPassCtx & {
  trace: TraceLog;
  errors: CompilationError[];
  failed: Set<string>;
  resilient: boolean;
};

export class LirPassManager {
  passes: LirFuncPass[];
  trace: TraceLog | null;
  checkEachPass: boolean;

  constructor() {
    this.passes = [];
    this.trace = null;
    this.checkEachPass = false;
  }

  addPass(pass: LirFuncPass): void {
    this.passes.push(pass);
  }

  setTrace(trace: TraceLog | null): void {
    this.trace = trace;
  }

  setCheckEachPass(enabled: boolean): void {
    this.checkEachPass = enabled;
  }

  run(funcs: LIRFunc[], ctx: LirRunCtx): LIRFunc[] {
    for (const pass of this.passes) {
      this._runPass(pass, funcs, ctx);
    }
    return funcs;
  }

  _runPass(pass: LirFuncPass, funcs: LIRFunc[], ctx: LirRunCtx): void {
    const trace = ctx.trace;
    pass.trace = trace;
    trace.phaseStart(pass.phase);
    const t0 = performance.now();

    pass.begin(ctx);
    for (let i = 0; i < funcs.length; i++) {
      const func = funcs[i];
      if (ctx.failed.has(func.name)) continue;
      try {
        const out = pass.run(func, ctx);
        if (out && out !== func) funcs[i] = out;
      } catch (e) {
        const msg = (e as Error).message;
        ctx.errors.push(new CompilationError(pass.phase, func.name, msg));
        ctx.failed.add(func.name);
        trace.errorEvent(pass.phase, func.name, msg);
        if (!ctx.resilient) break;
      }
    }
    pass.end(ctx);

    trace.phaseEnd(pass.phase, performance.now() - t0);
    if (this.checkEachPass) this._verifyFuncs(funcs, ctx, pass.name);
    pass.trace = null;
  }

  _verifyFuncs(funcs: readonly LIRFunc[], ctx: LirRunCtx, passName: string): void {
    for (const func of funcs) {
      if (ctx.failed.has(func.name)) continue;
      const err = checkIRInvariants(IRLevel.LIR, func, func.name, passName);
      if (!err) continue;
      ctx.errors.push(err);
      ctx.failed.add(func.name);
      ctx.trace.errorEvent('verification', func.name, err.message);
      if (!ctx.resilient) throw new Error(err.toString());
    }
  }
}
