import { PrimFuncPass } from '../tir_pass.js';
import { MemoryPlanner } from './memory_planning.js';
import { FuncAttr } from '../../ir/func_attrs.js';
import { explainer } from '../explain.js';
import type { PrimFunc } from '../../ir/tensor/nodes.js';
import type { TirPassCtx } from '../tir_pass.js';
import type { CompilerConfig, CompileTarget } from '../../support/config_types.js';

type MemoryPlanConfig = CompilerConfig & { target?: CompileTarget & { cacheLineSizeBytes?: number } };

export class MemoryPlanPass extends PrimFuncPass {
  config: MemoryPlanConfig;
  private _planner: MemoryPlanner | null;

  constructor(config: MemoryPlanConfig) {
    super('MemoryPlanPass', 'memoryPlanning');
    this.config = config;
    this._planner = null;
  }

  override begin(ctx: TirPassCtx): void {
    const alignment = this.config.memory.alignment || this.config.target?.cacheLineSizeBytes || 64;
    this._planner = new MemoryPlanner({
      alignment,
      enableInplace: this.config.memory.inplaceReuse,
      allocStrategy: this.config.memory.allocStrategy,
      poolAllocation: this.config.memory.poolAllocation,
    });
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): void {
    if (pf.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)) return;
    const ft0 = performance.now();
    const { plan } = (this._planner as MemoryPlanner).planAndRewrite(pf);
    const report = plan.getReport();
    ctx.trace.memoryStats(pf.name, {
      durationMs: performance.now() - ft0,
      peakMemory: report.peakMemory,
      totalTemporaries: report.totalTemporaries,
      totalInplace: report.totalInplace,
    });

    ctx.trace.memoryPlan(pf.name, plan.lifetimes());

    const explain = explainer(ctx.trace, this.name);
    if (explain) {
      explain(pf.name, `${report.totalTemporaries} temporaries fit in ${report.peakMemory} bytes`,
        'buffers whose live ranges never overlap can share one slot, so the function allocates its peak rather than its total',
        { peakMemory: report.peakMemory, totalTemporaries: report.totalTemporaries, reusedInPlace: report.totalInplace });
    }
  }
}
