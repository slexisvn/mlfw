import { PrimFuncPass } from '../tir_pass.js';
import { MemoryPlanner } from './memory_planning.js';
import { FuncAttr } from '../../ir/func_attrs.js';

export class MemoryPlanPass extends PrimFuncPass {
  constructor(config) {
    super('MemoryPlanPass', 'memoryPlanning');
    this.config = config;
    this._planner = null;
  }

  begin(ctx) {
    const alignment = this.config.memory.alignment || this.config.target?.cacheLineSizeBytes || 64;
    this._planner = new MemoryPlanner({
      alignment,
      enableInplace: this.config.memory.inplaceReuse,
      allocStrategy: this.config.memory.allocStrategy,
      poolAllocation: this.config.memory.poolAllocation,
    });
  }

  run(pf, ctx) {
    if (pf.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)) return;
    const ft0 = performance.now();
    const { plan } = this._planner.planAndRewrite(pf);
    const report = plan.getReport();
    ctx.trace.memoryStats(pf.name, {
      durationMs: performance.now() - ft0,
      peakMemory: report.peakMemory,
      totalTemporaries: report.totalTemporaries,
      totalInplace: report.totalInplace,
    });
  }
}
