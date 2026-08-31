import { checkIRInvariants } from '../support/invariant_check.js';
import type { CompilationError } from '../support/trace.js';
import type { IRLevelValue } from '../ir/verify.js';
import type { TraceLog } from '../support/trace.js';
import type { InstrumentedPass, PassInstrument } from '../support/pass_instrument.js';
import type { PassContext, PassLike, PassResultValue } from './pass.js';

export type NamedFunc = { name: string };
export type SkipCtx = { passContext?: PassContext | null; trace?: TraceLog | null };
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
  instruments: PassInstrument[];

  protected abstract readonly irLevel: IRLevelValue;

  constructor() {
    this.passes = [];
    this.trace = null;
    this.checkEachPass = false;
    this.instruments = [];
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

  addInstrument(instrument: PassInstrument): void {
    this.instruments.push(instrument);
  }

  protected _skipped(pass: PassLike, ctx: SkipCtx): boolean {
    const passContext = ctx.passContext;
    if (!passContext || passContext.shouldRun(pass)) return false;
    const trace = ctx.trace || this.trace;
    if (trace) trace.passSkipped(pass.name, this.irLevel);
    return true;
  }

  protected _notifyBefore(pass: InstrumentedPass, target: unknown): void {
    const instruments = this.instruments;
    for (let i = 0; i < instruments.length; i++) {
      const hook = instruments[i].runBeforePass;
      if (hook) hook.call(instruments[i], pass, target, this.irLevel);
    }
  }

  protected _notifyAfter(pass: InstrumentedPass, target: unknown, result: PassResultValue | null): void {
    const instruments = this.instruments;
    for (let i = 0; i < instruments.length; i++) {
      const hook = instruments[i].runAfterPass;
      if (hook) hook.call(instruments[i], pass, target, this.irLevel, result);
    }
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
