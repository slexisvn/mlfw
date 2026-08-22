import { PrimFuncPass } from '../tir_pass.js';
import { Schedule } from '../../schedule/schedule.js';
import { SchedulePolicy } from '../../schedule/rules.js';
import { Autotuner } from '../../autotune/autotuner.js';
import { applyDeterministicGpuSchedule } from '../../schedule/gpu_matmul_schedule.js';
import { FuncAttr } from '../../ir/func_attrs.js';
import type { PrimFunc } from '../../ir/tensor/nodes.js';
import type { TirPassCtx } from '../tir_pass.js';
import type { CompilerConfig, CompileTarget } from '../../pipeline/pipeline_types.js';

export class SchedulePass extends PrimFuncPass {
  config: CompilerConfig;
  target: CompileTarget;
  private _autotuner: Autotuner | null;
  private _policy: SchedulePolicy | null;

  constructor(config: CompilerConfig) {
    super('SchedulePass', 'scheduling');
    this.config = config;
    this.target = config.target;
    this.snapshotPoint = 'afterScheduling';
    this._autotuner = null;
    this._policy = null;
  }

  override begin(ctx: TirPassCtx): void {
    const sCfg = this.config.scheduling as Record<string, boolean>;
    if (sCfg.autotune) {
      this._autotuner = new Autotuner(this.target, { ...sCfg, fastMath: !!(this.config.optimization as { fastMath?: boolean } | undefined)?.fastMath }, ctx.trace);
    } else if (sCfg.enabled) {
      this._policy = new SchedulePolicy(this.target, null, ctx.trace);
    }
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): void {
    const sCfg = this.config.scheduling as Record<string, boolean>;
    const trace = ctx.trace;
    if (sCfg.autotune) {
      if (pf.hasAttr(FuncAttr.EXTERNAL_CODEGEN) || pf.hasAttr(FuncAttr.TENSOR_INTRIN)) return;
      const ft0 = performance.now();
      const tuneResult = (this._autotuner as Autotuner).tuneAndApply(pf);
      const durationMs = performance.now() - ft0;
      let cacheHits = 0, blockCount = 0;
      if (tuneResult && tuneResult.results) {
        blockCount = tuneResult.results.size;
        for (const [blockName, r] of tuneResult.results) {
          if (r.fromCache) cacheHits++;
          if (trace.explainsEnabled) {
            trace.explain('schedule', blockName, r.sketchName,
              `autotuned: best of search${r.fromCache ? ' (cached)' : ''}, score ${r.score != null ? r.score.toFixed(3) : 'n/a'}`,
              { target: this.target.name, params: r.params });
          }
        }
      }
      trace.autotuneStats(pf.name, { durationMs, blockCount, applied: !!(tuneResult && tuneResult.applied), cacheHits });
    } else if (sCfg.enabled || sCfg.gpuTiling) {
      if (pf.hasAttr(FuncAttr.EXTERNAL_CODEGEN) || pf.hasAttr(FuncAttr.TENSOR_INTRIN)) return;
      const ft0 = performance.now();
      const sch = new Schedule(pf);
      const handled = applyDeterministicGpuSchedule(sch, this.target as never, sCfg as never);
      if (!handled && sCfg.enabled) {
        (this._policy as SchedulePolicy).applyToAllBlocks(sch);
      }
      trace.functionEvent('scheduling', pf.name, { durationMs: performance.now() - ft0 });
    }
  }
}
