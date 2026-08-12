import { PrimFuncPass } from '../tir_pass.js';
import { simplifyPrimFunc } from './simplify_tir.js';
import type { PrimFunc } from '../../ir/tensor/nodes.js';
import type { TirPassCtx } from '../tir_pass.js';

export class SimplifyPass extends PrimFuncPass {
  constructor() {
    super('SimplifyPass', 'simplify');
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): PrimFunc {
    const ft0 = performance.now();
    simplifyPrimFunc(pf);
    ctx.trace.functionEvent('simplify', pf.name, { durationMs: performance.now() - ft0 });
    return pf;
  }
}
