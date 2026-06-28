import { PrimFuncPass } from '../tir_pass.js';
import { simplifyPrimFunc } from './simplify_tir.js';

export class SimplifyPass extends PrimFuncPass {
  constructor() {
    super('SimplifyPass', 'simplify');
  }

  run(pf, ctx) {
    const ft0 = performance.now();
    simplifyPrimFunc(pf);
    ctx.trace.functionEvent('simplify', pf.name, { durationMs: performance.now() - ft0 });
    return pf;
  }
}
