import { PrimFuncPass } from '../tir_pass.js';
import { Schedule } from '../../schedule/schedule.js';
import { SchedulePolicy } from '../../schedule/rules.js';
import { Autotuner } from '../../autotune/autotuner.js';
import { applyDeterministicGpuMatmul, applyDeterministicGpuConv } from '../../schedule/gpu_matmul_schedule.js';
import { applyImplicitGemmConv } from '../../schedule/conv_implicit_gemm.js';

export class SchedulePass extends PrimFuncPass {
  constructor(config) {
    super('SchedulePass', 'scheduling');
    this.config = config;
    this.target = config.target;
    this.snapshotPoint = 'afterScheduling';
    this._autotuner = null;
    this._policy = null;
  }

  begin(ctx) {
    const sCfg = this.config.scheduling;
    if (sCfg.autotune) {
      this._autotuner = new Autotuner(this.target, sCfg, ctx.trace);
    } else if (sCfg.enabled) {
      this._policy = new SchedulePolicy(this.target, null, ctx.trace);
    }
  }

  run(pf, ctx) {
    const sCfg = this.config.scheduling;
    const trace = ctx.trace;
    if (sCfg.autotune) {
      if (pf.cublasInfo || pf._tensorIntrin) return;
      const ft0 = performance.now();
      const tuneResult = this._autotuner.tuneAndApply(pf);
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
      if (pf.cublasInfo || pf._tensorIntrin) return;
      const ft0 = performance.now();
      const sch = new Schedule(pf);
      let handled = false;
      if (this.target.isGPU() && !this.target.isWebGPU()) {
        handled = applyDeterministicGpuMatmul(sch, this.target, sCfg);
        if (!handled) handled = applyImplicitGemmConv(sch, this.target, sCfg);
        if (!handled) handled = applyDeterministicGpuConv(sch, this.target);
      }
      if (!handled && sCfg.enabled) {
        this._policy.applyToAllBlocks(sch);
      }
      trace.functionEvent('scheduling', pf.name, { durationMs: performance.now() - ft0 });
    }
  }
}
