import { LirFuncPass } from '../lir_pass.js';
import { simplifyLirFunc } from './simplify_tir.js';
import type { LIRFunc } from '../../ir/lir/nodes.js';
import type { LirPassCtx } from '../lir_pass.js';

export class FlatIndexSimplifyPass extends LirFuncPass {
  constructor() {
    super('FlatIndexSimplifyPass', 'lirSimplify');
  }

  override run(func: LIRFunc, ctx: LirPassCtx): LIRFunc {
    const t0 = performance.now();
    simplifyLirFunc(func);
    ctx.trace.functionEvent('lirSimplify', func.name, { durationMs: performance.now() - t0 });
    return func;
  }
}
