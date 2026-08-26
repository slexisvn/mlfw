import { Schedule } from '../schedule/schedule.js';
import { AnalyticalCostModel, LearnedCostModel, GuidedCostModel } from './cost_model.js';
import { getSketchesForBlock } from './search_space.js';
import { TuningRecord, TuningDatabase } from './tuning_db.js';
import { BenchmarkRunner } from './benchmark.js';
import { Deadline } from './budget.js';
import { buildBlockMap, computeWorkloadKey } from './workload_key.js';
import { collectAllBlockNames } from './block_analysis.js';
import { buildBlockDAG, findFusibleConsumer } from './block_dag.js';
import { classifyBlock, SchedulePolicy } from '../schedule/rules.js';
import { applyDeterministicGpuSchedule } from '../schedule/gpu_matmul_schedule.js';
import { BlockTuningSession, gpuThreadBlockSize } from './session.js';
import { clonePrimFunc } from './tune_ir.js';
import { ForKind } from '../ir/tensor/nodes.js';
import { TaskScheduler } from './task_scheduler.js';
import { getMeasurer } from '../../runtime/measurer_registry.js';
import { FuncAttr } from '../ir/func_attrs.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';
import type { CostModelTarget } from './cost_model.js';
import type { MeasurerLike } from './benchmark.js';
import type { SchedulerPolicy, TuningTask } from './task_scheduler.js';
import type { SketchParams, ScheduleSketch } from './sketch.js';
import { TraceLevel } from '../pipeline/trace.js';
import type { TraceLog } from '../pipeline/trace.js';
import type { RoundReport } from './session.js';

export type AutotuneWarning = { stage: string; func: string | null; block: string | null; message: string; error: unknown };
export type AutotunerTarget = CostModelTarget;
export type TuneResult = {
  blockName?: string;
  sketchName: string;
  params: SketchParams;
  score: number;
  fromCache?: boolean;
  medianMs?: number | null;
  minMs?: number | null;
};

export type TuneResults = Map<string, TuneResult>;
export type ApplyResult = { func: PrimFunc; results: TuneResults; applied: boolean };

export type TuneTaskKind = 'cache' | 'empty' | 'session';
export type TuneTask = Partial<TuningTask> & {
  key: string;
  kind: TuneTaskKind;
  weight: number;
  cached?: TuningRecord | null;
  session?: BlockTuningSession;
  stored?: boolean;
};

export type AutotuneConfigOpts = Readonly<{
  strategy?: string;
  numTrials?: number;
  populationSize?: number;
  numGenerations?: number;
  mutationRate?: number;
  eliteRatio?: number;
  seed?: number;
  timeBudgetMs?: number;
  clock?: (() => number) | null;
  tuningDB?: TuningDatabase | null;
  useTuningDB?: boolean;
  measurer?: MeasurerLike | null;
  hardwareMeasure?: boolean;
  enableBenchmark?: boolean;
  benchmarkWarmup?: number;
  benchmarkRepeat?: number;
  benchmarkMaxCv?: number;
  topKForBenchmark?: number;
  fastMath?: boolean;
  maxRoundsPerTask?: number;
  plateauPatience?: number;
  schedulerPolicy?: SchedulerPolicy | null;
  onWarning?: ((w: AutotuneWarning) => void) | null;
  richGpu?: boolean;
}>;

function resolveMeasurer(target: ScheduleTarget): MeasurerLike | null {
  if (target.isCPU()) return null;
  const measurer = getMeasurer(target.kind as string) as unknown as MeasurerLike | null;
  if (!measurer) {
    throw new Error('hardwareMeasure requested for target \'' + target.kind + '\' but no measurer is registered for it; the corresponding runtime must be loaded (Node: import \'#io/cuda_runtime\') before compiling');
  }
  return measurer;
}

class AutotuneConfig {
  strategy: string;
  numTrials: number;
  populationSize: number;
  numGenerations: number;
  mutationRate: number | undefined;
  eliteRatio: number | undefined;
  seed: number;
  timeBudgetMs: number;
  clock: (() => number) | null;
  tuningDB: TuningDatabase | null;
  useTuningDB: boolean;
  measurer: MeasurerLike | null;
  hardwareMeasure: boolean;
  enableBenchmark: boolean;
  benchmarkWarmup: number;
  benchmarkRepeat: number;
  benchmarkMaxCv: number;
  topKForBenchmark: number;
  numericMode: string;
  maxRoundsPerTask: number;
  plateauPatience: number;
  schedulerPolicy: SchedulerPolicy | null;
  onWarning: ((w: AutotuneWarning) => void) | null;
  richGpu?: boolean;
  topKForBenchmark2?: never;
  [key: string]: unknown;

  constructor(opts: AutotuneConfigOpts = {}) {
    this.strategy = opts.strategy || 'evolutionary';
    this.numTrials = opts.numTrials || 64;
    this.populationSize = opts.populationSize || 32;
    this.numGenerations = opts.numGenerations || 10;
    this.mutationRate = opts.mutationRate;
    this.eliteRatio = opts.eliteRatio;
    this.seed = opts.seed || 42;
    this.timeBudgetMs = opts.timeBudgetMs || 30000;
    this.clock = opts.clock || null;
    this.tuningDB = opts.tuningDB || null;
    this.useTuningDB = opts.useTuningDB !== false;
    this.measurer = opts.measurer || null;
    this.hardwareMeasure = opts.hardwareMeasure ?? false;
    this.enableBenchmark = opts.enableBenchmark ?? (this.hardwareMeasure || !!opts.measurer);
    this.benchmarkWarmup = opts.benchmarkWarmup ?? 3;
    this.benchmarkRepeat = opts.benchmarkRepeat ?? 10;
    this.benchmarkMaxCv = opts.benchmarkMaxCv ?? 0;
    this.topKForBenchmark = opts.topKForBenchmark ?? 5;
    this.numericMode = opts.fastMath ? 'n3' : 'n1';
    this.maxRoundsPerTask = opts.maxRoundsPerTask ?? 8;
    this.plateauPatience = opts.plateauPatience ?? 2;
    this.schedulerPolicy = opts.schedulerPolicy || null;
    this.onWarning = opts.onWarning || null;
  }
}

export class Autotuner {
  target: AutotunerTarget;
  config: AutotuneConfig;
  trace: TraceLog | null;
  analyticalModel: AnalyticalCostModel;
  learnedModel: LearnedCostModel;
  costModel: GuidedCostModel;
  db: TuningDatabase;
  benchmarkRunner: BenchmarkRunner | null;
  scheduler: TaskScheduler;
  private _funcName: string | null;

  constructor(target: AutotunerTarget, config: AutotuneConfig | AutotuneConfigOpts = {}, trace: TraceLog | null = null) {
    this.target = target;
    this.config = config instanceof AutotuneConfig ? config : new AutotuneConfig(config);
    this.trace = trace;
    this._funcName = null;
    if (this.config.hardwareMeasure) this.config.measurer = resolveMeasurer(target);
    this.analyticalModel = new AnalyticalCostModel(target);
    this.learnedModel = new LearnedCostModel();
    this.costModel = new GuidedCostModel(this.analyticalModel, this.learnedModel);
    this.db = this.config.tuningDB instanceof TuningDatabase ? this.config.tuningDB : new TuningDatabase();
    const warn = (stage: string, block: string | null, e: unknown) => this._warn(stage, block, e);
    this.benchmarkRunner = this.config.enableBenchmark
      ? new BenchmarkRunner(target, {
          warmup: this.config.benchmarkWarmup,
          repeat: this.config.benchmarkRepeat,
          maxCv: this.config.benchmarkMaxCv,
          measurer: this.config.measurer,
          warn
        })
      : null;
    this.scheduler = new TaskScheduler(this.config.schedulerPolicy);
  }

  _warn(stage: string, blockName: string | null, error: unknown): void {
    const message = error && (error as Error).message ? (error as Error).message : String(error);
    if (this.config.onWarning) {
      try {
        this.config.onWarning({ stage, func: this._funcName, block: blockName || null, message, error });
      } catch (e) { void e; }
    }
    if (this.trace) {
      this.trace.warn('autotune', this._funcName as string, `${stage}${blockName ? ' [' + blockName + ']' : ''}: ${message}`);
    }
  }

  tune(primFunc: PrimFunc, blockName: string | null = null): TuneResults {
    this._funcName = primFunc.name;
    const blockNames = blockName ? [blockName] : collectAllBlockNames(primFunc.body);
    const blockMap = buildBlockMap(primFunc.body);
    const dag = buildBlockDAG(primFunc);
    const deadline = new Deadline(this.config.timeBudgetMs, this.config.clock);

    const tasksByKey = new Map<string, TuneTask>();
    const keyByBlock = new Map<string, string>();

    for (const name of blockNames) {
      const key = computeWorkloadKey(primFunc, name, this.target, blockMap, this.config.numericMode);
      keyByBlock.set(name, key);
      const existing = tasksByKey.get(key);
      if (existing) { existing.weight++; continue; }

      if (this.config.useTuningDB && this.db.has(key)) {
        tasksByKey.set(key, { key, kind: 'cache', cached: this.db.lookup(key), weight: 1 });
        continue;
      }

      const sketches = getSketchesForBlock(primFunc, name, this.target, blockMap, { richGpu: (this.config.richGpu as boolean | undefined) ?? !!this.config.measurer, dag });
      if (sketches.length === 0) {
        tasksByKey.set(key, { key, kind: 'empty', weight: 1 });
        continue;
      }

      const session = new BlockTuningSession({
        target: this.target, primFunc, blockName: name, blockMap, sketches,
        costModel: this.costModel, learnedModel: this.learnedModel,
        benchmarkRunner: this.benchmarkRunner, config: this.config, deadline,
        warn: (stage: string, block: string | null, e: unknown) => this._warn(stage, block, e),
        onRound: this.trace ? (report: RoundReport) => this._reportRound(primFunc.name, report) : null
      });
      tasksByKey.set(key, { key, kind: 'session', session, weight: 1 });
    }

    const sessionTasks = [...tasksByKey.values()].filter(t => t.kind === 'session') as unknown as TuningTask[];
    if (sessionTasks.length > 0) this.scheduler.run(sessionTasks, deadline, this.config);

    const results = new Map<string, TuneResult>();
    for (const name of blockNames) {
      const task = tasksByKey.get(keyByBlock.get(name) as string) as TuneTask;
      if (task.kind === 'cache') {
        const cached = task.cached as TuningRecord;
        results.set(name, { blockName: name, sketchName: cached.sketchName, params: cached.params, score: cached.score, fromCache: true });
        continue;
      }
      if (task.kind === 'empty') continue;

      const best = (task.session as BlockTuningSession).best();
      if (!best) continue;

      if (this.config.useTuningDB && !task.stored) {
        const record = new TuningRecord(task.key, best.sketchName, best.params, best.score, (task.session as BlockTuningSession).bestTrace(), this.db.version);
        record.medianMs = best.medianMs || null;
        record.minMs = best.minMs || null;
        this.db.store(task.key, record);
        task.stored = true;
      }

      results.set(name, { blockName: name, sketchName: best.sketchName, params: best.params, score: best.score, fromCache: false, medianMs: best.medianMs, minMs: best.minMs });
    }

    return results;
  }

  tuneAndApply(primFunc: PrimFunc, blockName: string | null = null): ApplyResult {
    const tuneResults = this.tune(primFunc, blockName);
    if (tuneResults.size > 0) {
      const best = this._applyBestSchedule(primFunc, tuneResults);
      if (best) return { func: best.func, results: tuneResults, applied: true };
    }
    return { func: primFunc, results: tuneResults, applied: false };
  }

  _applyBestSchedule(primFunc: PrimFunc, tuneResults: TuneResults): { func: PrimFunc } | null {
    const baseline = this._buildDefaultSchedule(primFunc);
    const baselineValid = !!baseline && this._scheduleIsValid(baseline);
    const baselineStrong = baselineValid && this._isStrongBackendSchedule(baseline);

    const tuned = this._buildTunedSchedule(primFunc, tuneResults);
    const tunedValid = !!tuned && this._scheduleIsValid(tuned);
    const tunedStrong = tunedValid && this._isStrongBackendSchedule(tuned);

    const mayDisplaceBaseline = !baselineStrong || (tunedStrong && this.config.measurer != null);
    if (tunedValid && mayDisplaceBaseline) {
      this._adoptSchedule(primFunc, tuned);
      return { func: primFunc };
    }

    if (tuned && !tunedValid) {
      this._warn('tuned-schedule-invalid', null, new Error('tuned schedule exceeds target thread-block limit; falling back to default'));
    } else if (tunedValid && baselineStrong) {
      this._warn('baseline-preferred', null, new Error('cost-model-only tuning cannot displace the deterministic GPU schedule without hardware measurement; keeping the deterministic kernel'));
    }

    if (baselineValid) {
      this._adoptSchedule(primFunc, baseline);
      return { func: primFunc };
    }
    if (tunedValid) {
      this._adoptSchedule(primFunc, tuned);
      return { func: primFunc };
    }
    this._warn('no-valid-schedule', null, new Error('neither tuned nor default schedule is valid; leaving function unscheduled'));
    return null;
  }

  _isStrongBackendSchedule(func: PrimFunc): boolean {
    return !!func && func.getAttr(FuncAttr.GPU_REGISTER_BLOCKED) === true;
  }

  _buildTunedSchedule(primFunc: PrimFunc, tuneResults: TuneResults): PrimFunc | null {
    try {
      const work = clonePrimFunc(primFunc);
      const sch = new Schedule(work);
      const blockMap = buildBlockMap(work.body);
      const dag = buildBlockDAG(work);

      const fusedAway = new Set<string>();
      const ordered = [];
      for (const entry of tuneResults) {
        if (entry[1].sketchName === 'fused') {
          const consumer = findFusibleConsumer(work, dag, entry[0], classifyBlock);
          if (consumer) fusedAway.add(consumer);
          ordered.unshift(entry);
        } else {
          ordered.push(entry);
        }
      }

      const applied = new Set();
      for (const [blockName, result] of ordered) {
        if (fusedAway.has(blockName) || applied.has(result)) continue;
        applied.add(result);
        if (!result.sketchName || !result.params) continue;
        try {
          const sketches = getSketchesForBlock(work, blockName, this.target, blockMap, { richGpu: this.config.richGpu ?? !!this.config.measurer, dag });
          const sketch = sketches.find(s => s.name === result.sketchName);
          if (sketch && this._fitsThreadBlock(work, blockName, sketch, result.params)) {
            const apply = sketch.instantiate(result.params);
            apply(sch, blockName, this.target);
          }
        } catch (e) {
          this._warn('apply-tuned-block', blockName, e);
          continue;
        }
      }

      this._scheduleResidualBlocks(sch, fusedAway);
      if (this.trace) this.trace.scheduleTrace(primFunc.name, sch.trace.serialize());
      return work;
    } catch (e) {
      this._warn('build-tuned-schedule', null, e);
      return null;
    }
  }

  _scheduleResidualBlocks(sch: Schedule, fusedAway: ReadonlySet<string>): void {
    let policy = null;
    for (const name of collectAllBlockNames(sch.func.body)) {
      if (fusedAway.has(name) || this._blockIsParallelized(sch, name)) continue;
      if (!policy) policy = new SchedulePolicy(this.target);
      policy.applyToBlock(sch, name);
    }
  }

  _blockIsParallelized(sch: Schedule, blockName: string): boolean {
    let loops;
    try { loops = sch.getLoops(blockName); } catch (e) { this._warn('block-loops', blockName, e); return true; }
    for (const l of loops) {
      if (l.kind === ForKind.THREAD_BINDING || l.kind === ForKind.PARALLEL || l.kind === ForKind.VECTORIZED) return true;
    }
    return false;
  }

  _scheduleIsValid(func: PrimFunc): boolean {
    if (!this.target.isGPU || !this.target.isGPU() || !this.target.maxThreadsPerBlock) return true;
    return gpuThreadBlockSize(func) <= this.target.maxThreadsPerBlock;
  }

  _buildDefaultSchedule(primFunc: PrimFunc): PrimFunc | null {
    try {
      const work = clonePrimFunc(primFunc);
      const sch = new Schedule(work);
      if (!applyDeterministicGpuSchedule(sch, this.target, this.config as never)) {
        new SchedulePolicy(this.target).applyToAllBlocks(sch);
      }
      return work;
    } catch (e) {
      this._warn('build-default-schedule', null, e);
      return null;
    }
  }

  _reportRound(funcName: string, report: RoundReport): void {
    (this.trace as TraceLog).emit({
      type: 'autotune_round',
      funcName,
      blockName: report.blockName,
      round: report.round,
      measured: report.measured,
      scores: report.candidates.map(candidate => ({ sketch: candidate.sketchName, score: candidate.score })),
      bestSketch: report.best ? report.best.sketchName : null,
      bestScore: report.best ? report.best.score : null,
      level: TraceLevel.DEBUG,
    });
  }

  _adoptSchedule(target: PrimFunc, src: PrimFunc): void {
    Object.assign(target, src);
    target._setChild('body', target.body);
  }

  _fitsThreadBlock(primFunc: PrimFunc, blockName: string, sketch: ScheduleSketch, params: SketchParams): boolean {
    if (!this.target.isGPU || !this.target.isGPU() || !this.target.maxThreadsPerBlock) return true;
    try {
      const trial = clonePrimFunc(primFunc);
      sketch.instantiate(params)(new Schedule(trial), blockName, this.target);
      return gpuThreadBlockSize(trial) <= this.target.maxThreadsPerBlock;
    } catch (e) {
      this._warn('fits-thread-block', blockName, e);
      return false;
    }
  }
}
