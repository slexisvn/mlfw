import { PrimFuncPass } from '../tir_pass.js';
import { simplifyAndReport, simplifyPrimFunc } from './simplify_tir.js';
import type { PrimFunc } from '../../ir/tensor/nodes.js';
import type { TirPassCtx } from '../tir_pass.js';

export class SimplifyPass extends PrimFuncPass {
  constructor() {
    super('SimplifyPass', 'simplify');
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): PrimFunc {
    return simplifyAndReport(this, pf, simplifyPrimFunc, ctx.trace);
  }
}
