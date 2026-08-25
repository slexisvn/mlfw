import { LirFuncPass } from '../lir_pass.js';
import { simplifyAndReport, simplifyLirFunc } from './simplify_tir.js';
import type { LIRFunc } from '../../ir/lir/nodes.js';
import type { LirPassCtx } from '../lir_pass.js';

export class FlatIndexSimplifyPass extends LirFuncPass {
  constructor() {
    super('FlatIndexSimplifyPass', 'lirSimplify');
  }

  override run(func: LIRFunc, ctx: LirPassCtx): LIRFunc {
    return simplifyAndReport(this, func, simplifyLirFunc, ctx.trace);
  }
}
