import { CompilationError } from '../pipeline/trace.js';
import { printTensorIR } from '../ir/tensor/printer.js';
import { PassManagerBase } from './pass_manager_base.js';
import { PassResult } from './pass.js';
import { IRLevel } from '../ir/verify.js';
import type { TirModule } from '../ir/tensor/module.js';
import type { PrimFuncPass, TirModulePass, TirPassCtx } from './tir_pass.js';
import type { PassResultValue } from './pass.js';
import type { TraceLog } from '../pipeline/trace.js';

export type TirPassAny = PrimFuncPass | TirModulePass;
export type TirRunCtx = TirPassCtx & {
  trace: TraceLog;
  errors: CompilationError[];
  failed: Set<string>;
  resilient: boolean;
};

export class TirPassManager extends PassManagerBase<TirPassAny> {
  protected override readonly irLevel = IRLevel.TIR;

  run(module: TirModule, ctx: TirRunCtx): TirModule {
    for (const pass of this.passes) {
      this._runPass(pass, module, ctx);
    }
    return module;
  }

  _runPass(pass: TirPassAny, module: TirModule, ctx: TirRunCtx): void {
    const trace = ctx.trace;
    pass.trace = trace;
    trace.phaseStart(pass.phase);
    const t0 = performance.now();

    this._notifyBefore(pass, module);
    let result: PassResultValue | null = null;
    try {
      pass.begin(ctx);
      if ((pass as TirModulePass).runModule) this._runModulePass(pass as TirModulePass, module, ctx);
      else this._runFunctionPass(pass as PrimFuncPass, module, ctx);
      pass.end(ctx);
    } catch (e) {
      result = PassResult.FAILED;
      throw e;
    } finally {
      this._notifyAfter(pass, module, result);
    }

    trace.phaseEnd(pass.phase, performance.now() - t0);

    if (pass.snapshotPoint && trace.shouldSnapshot(pass.snapshotPoint as never)) {
      for (const pf of module) {
        if (!ctx.failed.has(pf.name)) trace.irDump(pass.snapshotPoint + ':' + pf.name, printTensorIR(pf));
      }
    }

    if (this.checkEachPass) this._verifyFuncs(IRLevel.TIR, module, ctx, pass.name);
    pass.trace = null;
  }

  _runModulePass(pass: TirModulePass, module: TirModule, ctx: TirRunCtx): void {
    try {
      pass.runModule(module, ctx);
    } catch (e) {
      const msg = (e as Error).message;
      ctx.errors.push(new CompilationError(pass.phase, module.name, msg));
      ctx.trace.errorEvent(pass.phase, module.name, msg);
      if (!ctx.resilient) throw e;
    }
  }

  _runFunctionPass(pass: PrimFuncPass, module: TirModule, ctx: TirRunCtx): void {
    for (const pf of [...module]) {
      if (ctx.failed.has(pf.name)) continue;
      try {
        const out = pass.run(pf, ctx);
        if (out && out !== pf) module.replaceFunction(pf.name, out);
      } catch (e) {
        const msg = (e as Error).message;
        ctx.errors.push(new CompilationError(pass.phase, pf.name, msg));
        ctx.failed.add(pf.name);
        ctx.trace.errorEvent(pass.phase, pf.name, msg);
        if (!ctx.resilient) break;
      }
    }
  }

}
