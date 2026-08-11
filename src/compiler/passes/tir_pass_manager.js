import { CompilationError } from '../pipeline/trace.js';
import { printTensorIR } from '../ir/tensor/printer.js';
import { checkIRInvariants } from '../pipeline/invariant_check.js';
import { IRLevel } from '../ir/verify.js';

export class TirPassManager {
  constructor() {
    this.passes = [];
    this.trace = null;
    this.checkEachPass = false;
  }

  addPass(pass) {
    this.passes.push(pass);
  }

  setTrace(trace) {
    this.trace = trace;
  }

  setCheckEachPass(enabled) {
    this.checkEachPass = enabled;
  }

  run(module, ctx) {
    for (const pass of this.passes) {
      this._runPass(pass, module, ctx);
    }
    return module;
  }

  _runPass(pass, module, ctx) {
    const trace = ctx.trace;
    pass.trace = trace;
    trace.phaseStart(pass.phase);
    const t0 = performance.now();

    pass.begin(ctx);
    if (pass.runModule) this._runModulePass(pass, module, ctx);
    else this._runFunctionPass(pass, module, ctx);
    pass.end(ctx);

    trace.phaseEnd(pass.phase, performance.now() - t0);

    if (pass.snapshotPoint && trace.shouldSnapshot(pass.snapshotPoint)) {
      for (const pf of module) {
        if (!ctx.failed.has(pf.name)) trace.irDump(pass.snapshotPoint + ':' + pf.name, printTensorIR(pf));
      }
    }

    if (this.checkEachPass) this._verifyFuncs(module, ctx, pass.name);
    pass.trace = null;
  }

  _runModulePass(pass, module, ctx) {
    try {
      pass.runModule(module, ctx);
    } catch (e) {
      ctx.errors.push(new CompilationError(pass.phase, module.name, e.message));
      ctx.trace.errorEvent(pass.phase, module.name, e.message);
      if (!ctx.resilient) throw e;
    }
  }

  _runFunctionPass(pass, module, ctx) {
    for (const pf of [...module]) {
      if (ctx.failed.has(pf.name)) continue;
      try {
        const out = pass.run(pf, ctx);
        if (out && out !== pf) module.replaceFunction(pf.name, out);
      } catch (e) {
        ctx.errors.push(new CompilationError(pass.phase, pf.name, e.message));
        ctx.failed.add(pf.name);
        ctx.trace.errorEvent(pass.phase, pf.name, e.message);
        if (!ctx.resilient) break;
      }
    }
  }

  _verifyFuncs(module, ctx, passName) {
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
