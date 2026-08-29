import { CompilationError } from '../pipeline/trace.js';
import { PassManagerBase } from './pass_manager_base.js';
import { PassResult } from './pass.js';
import { IRLevel } from '../ir/verify.js';
import type { LIRFunc } from '../ir/lir/nodes.js';
import type { LirFuncPass, LirPassCtx } from './lir_pass.js';
import type { PassContext, PassResultValue } from './pass.js';
import type { TraceLog } from '../pipeline/trace.js';

export type LirRunCtx = LirPassCtx & {
  trace: TraceLog;
  errors: CompilationError[];
  failed: Set<string>;
  resilient: boolean;
  passContext?: PassContext | null;
};

export class LirPassManager extends PassManagerBase<LirFuncPass> {
  protected override readonly irLevel = IRLevel.LIR;

  run(funcs: LIRFunc[], ctx: LirRunCtx): LIRFunc[] {
    for (const pass of this.passes) {
      if (this._skipped(pass, ctx)) continue;
      this._runPass(pass, funcs, ctx);
    }
    return funcs;
  }

  _runPass(pass: LirFuncPass, funcs: LIRFunc[], ctx: LirRunCtx): void {
    const trace = ctx.trace;
    pass.trace = trace;
    trace.phaseStart(pass.phase);
    const t0 = performance.now();

    this._notifyBefore(pass, funcs);
    let result: PassResultValue | null = null;
    try {
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
          result = PassResult.FAILED;
          if (!ctx.resilient) break;
        }
      }
      pass.end(ctx);
    } finally {
      this._notifyAfter(pass, funcs, result);
    }

    trace.phaseEnd(pass.phase, performance.now() - t0);
    if (this.checkEachPass) this._verifyFuncs(IRLevel.LIR, funcs, ctx, pass.name);
    pass.trace = null;
  }

}
