import { CompilationError } from '../pipeline/trace.js';
import { printTensorIR } from '../ir/tensor/printer.js';
import { checkIRInvariants } from '../pipeline/invariant_check.js';
import { IRLevel } from '../ir/verify.js';
import type { TirModule } from '../ir/tensor/module.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { PrimFuncPass, TirModulePass, TirPassCtx } from './tir_pass.js';
import type { TraceLog } from '../pipeline/trace.js';

export type TirPassAny = PrimFuncPass | TirModulePass;
export type TirRunCtx = TirPassCtx & {
  trace: TraceLog;
  errors: CompilationError[];
  failed: Set<string>;
  resilient: boolean;
};

export class TirPassManager {
  passes: TirPassAny[];
  trace: TraceLog | null;
  checkEachPass: boolean;

  constructor() {
    this.passes = [];
    this.trace = null;
    this.checkEachPass = false;
  }

  addPass(pass: TirPassAny): void {
    this.passes.push(pass);
  }

  setTrace(trace: TraceLog | null): void {
    this.trace = trace;
  }

  setCheckEachPass(enabled: boolean): void {
    this.checkEachPass = enabled;
  }

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

    pass.begin(ctx);
    if ((pass as TirModulePass).runModule) this._runModulePass(pass as TirModulePass, module, ctx);
    else this._runFunctionPass(pass as PrimFuncPass, module, ctx);
    pass.end(ctx);

    trace.phaseEnd(pass.phase, performance.now() - t0);

    if (pass.snapshotPoint && trace.shouldSnapshot(pass.snapshotPoint as never)) {
      for (const pf of module) {
        if (!ctx.failed.has(pf.name)) trace.irDump(pass.snapshotPoint + ':' + pf.name, printTensorIR(pf));
      }
    }

    if (this.checkEachPass) this._verifyFuncs(module, ctx, pass.name);
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

  _verifyFuncs(module: TirModule, ctx: TirRunCtx, passName: string): void {
    for (const pf of module) {
      if (ctx.failed.has(pf.name)) continue;
      const err = checkIRInvariants(IRLevel.TIR, pf, pf.name, passName);
      if (!err) continue;
      ctx.errors.push(err);
      ctx.failed.add(pf.name);
      ctx.trace.errorEvent('verification', pf.name, err.message);
      if (!ctx.resilient) throw new Error(err.toString());
    }
  }
}
