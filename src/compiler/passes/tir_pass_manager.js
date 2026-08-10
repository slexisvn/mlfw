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

  run(primFuncs, ctx) {
    for (const pass of this.passes) {
      this._runPass(pass, primFuncs, ctx);
    }
    return primFuncs;
  }

  _runPass(pass, primFuncs, ctx) {
    const trace = ctx.trace;
    pass.trace = trace;
    trace.phaseStart(pass.phase);
    const t0 = performance.now();

    pass.begin(ctx);
    for (let i = 0; i < primFuncs.length; i++) {
      const pf = primFuncs[i];
      if (ctx.failed.has(pf.name)) continue;
      try {
        const out = pass.run(pf, ctx);
        if (out && out !== pf) primFuncs[i] = out;
      } catch (e) {
        ctx.errors.push(new CompilationError(pass.phase, pf.name, e.message));
        ctx.failed.add(pf.name);
        trace.errorEvent(pass.phase, pf.name, e.message);
        if (!ctx.resilient) break;
      }
    }
    pass.end(ctx);

    trace.phaseEnd(pass.phase, performance.now() - t0);

    if (pass.snapshotPoint && trace.shouldSnapshot(pass.snapshotPoint)) {
      for (const pf of primFuncs) {
        if (!ctx.failed.has(pf.name)) trace.irDump(pass.snapshotPoint + ':' + pf.name, printTensorIR(pf));
      }
    }

    if (this.checkEachPass) this._verifyFuncs(primFuncs, ctx, pass.name);
    pass.trace = null;
  }

  _verifyFuncs(primFuncs, ctx, passName) {
    for (const pf of primFuncs) {
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
