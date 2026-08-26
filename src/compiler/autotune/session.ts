import { Schedule } from '../schedule/schedule.js';
import { ScheduleValidator } from '../schedule/validator.js';
import { FeatureExtractor } from './features.js';
import { clonePrimFunc, extractBlockMini } from './tune_ir.js';
import { createSearchStrategy } from './search.js';
import { LearnedCostModel, GuidedCostModel } from './cost_model.js';
import type { AnalyticalCostModel } from './cost_model.js';
import type { PrimFunc, BlockNode, ForNode, IntImmNode } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';
import type { ScheduleSketch, SketchParams } from './sketch.js';
import type { Deadline } from './budget.js';
import type { SerializedStep } from '../schedule/trace.js';
import type { PopulationMember } from './search.js';

export type ScoredCandidate = { sketchName: string; params: SketchParams; score: number };
export type TuningRecordDraft = ScoredCandidate & { measuredScore: number; medianMs?: number; minMs?: number };
export type BenchmarkResult = { medianMs: number; minMs: number };
export type BenchmarkRunnerLike = { run(primFunc: PrimFunc): BenchmarkResult | null };
export type CostModelLike = { score(primFunc: PrimFunc): number };
export type WarnFn = (phase: string, subject: string, err: unknown) => void;
export type TimedCandidate = ScoredCandidate & { medianMs?: number | null; minMs?: number | null };
export type RoundReport = {
  blockName: string;
  round: number;
  candidates: readonly ScoredCandidate[];
  best: TimedCandidate | null;
  measured: boolean;
};
export type RoundObserver = (report: RoundReport) => void;
export type EnumerableSketch = ScheduleSketch & { enumerate(): SketchParams[] };

export type SessionConfig = Readonly<{ topKForBenchmark: number; [key: string]: unknown }>;

export type SessionOpts = {
  target: ScheduleTarget;
  primFunc: PrimFunc;
  blockName: string;
  sketches: ScheduleSketch[];
  benchmarkRunner?: BenchmarkRunnerLike | null;
  config: SessionConfig;
  deadline?: Deadline | null;
  warn?: WarnFn;
  onRound?: RoundObserver | null;
  costModel: CostModelLike & { analytical?: AnalyticalCostModel };
  learnedModel?: LearnedCostModel | null;
  blockMap: ReadonlyMap<string, BlockNode>;
};

const THREAD_AXES = ['threadIdx.x', 'threadIdx.y', 'threadIdx.z'];

export function gpuThreadBlockSize(func: PrimFunc): number {
  const perAxis: Record<string, number> = { 'threadIdx.x': 1, 'threadIdx.y': 1, 'threadIdx.z': 1 };
  const seen = new Set<object>();
  const visit = (node: unknown): void => {
    if (!node || typeof node !== 'object' || seen.has(node)) return;
    seen.add(node as object);
    const f = node as ForNode;
    if (f.type === 'ForNode' && f.threadTag && perAxis[f.threadTag] !== undefined) {
      const extent = f.extent && f.extent.type === 'IntImmNode' ? (f.extent as IntImmNode).value : 1;
      if (extent > perAxis[f.threadTag]) perAxis[f.threadTag] = extent;
    }
    const slots = node as Record<string, unknown>;
    for (const key in slots) {
      const child = slots[key];
      if (Array.isArray(child)) child.forEach(visit);
      else if (child && typeof child === 'object') visit(child);
    }
  };
  visit(func.body);
  return THREAD_AXES.reduce((product, axis) => product * perAxis[axis], 1);
}

export class BlockTuningSession {
  target: ScheduleTarget;
  primFunc: PrimFunc;
  blockName: string;
  sketches: ScheduleSketch[];
  benchmarkRunner: BenchmarkRunnerLike | null;
  config: SessionConfig;
  deadline: Deadline | null;
  costModel: CostModelLike;
  learnedModel: LearnedCostModel | null;
  evalFunc: PrimFunc;
  evalBlockName: string;
  sketchByName: Map<string, ScheduleSketch>;
  enumSketch: EnumerableSketch | null;
  enumParams: SketchParams[] | null;
  strategy: ReturnType<typeof createSearchStrategy>;
  population: PopulationMember[] | null;
  plateaued: boolean;
  private _warn: WarnFn;
  private _onRound: RoundObserver | null;
  private _rounds: number;
  private _warnedEvalSketches: Set<string>;
  private _best: TuningRecordDraft | null;

  constructor(opts: SessionOpts) {
    this.target = opts.target;
    this.primFunc = opts.primFunc;
    this.blockName = opts.blockName;
    this.sketches = opts.sketches;
    this.benchmarkRunner = opts.benchmarkRunner || null;
    this.config = opts.config;
    this.deadline = opts.deadline || null;
    this._warn = opts.warn || (() => {});
    this._onRound = opts.onRound || null;
    this._rounds = 0;
    this._warnedEvalSketches = new Set();

    const needsWholeFunc = this.sketches.some(s => s.name === 'fused');
    if (needsWholeFunc) {
      this.learnedModel = new LearnedCostModel();
      this.costModel = new GuidedCostModel(opts.costModel.analytical as AnalyticalCostModel, this.learnedModel);
    } else {
      this.costModel = opts.costModel;
      this.learnedModel = opts.learnedModel ?? null;
    }
    const mini = needsWholeFunc ? null : extractBlockMini(opts.primFunc, opts.blockName, opts.blockMap);
    this.evalFunc = mini || opts.primFunc;
    this.evalBlockName = opts.blockName;

    this.sketchByName = new Map();
    for (const s of this.sketches) this.sketchByName.set(s.name, s);

    const first = this.sketches[0] as EnumerableSketch | undefined;
    const single = this.sketches.length === 1 && typeof first?.enumerate === 'function';
    this.enumSketch = single ? (first as EnumerableSketch) : null;
    this.enumParams = single ? (first as EnumerableSketch).enumerate() : null;

    this.strategy = createSearchStrategy({ ...this.config, deadline: this.deadline });
    this.population = null;
    this._best = null;
    this.plateaued = false;
  }

  runRound(): number {
    const prev = this._best ? this._best.measuredScore : -Infinity;
    const candidates = this._produceCandidates();
    if (candidates.length === 0) {
      this.plateaued = true;
      return 0;
    }
    if (this.benchmarkRunner) {
      this._measureAndLearn(candidates);
    } else {
      const top = candidates[0];
      this._consider({ sketchName: top.sketchName, params: top.params, score: top.score, measuredScore: top.score });
      this.plateaued = true;
    }
    const now = this._best ? this._best.measuredScore : -Infinity;
    if (this._onRound) {
      this._onRound({
        blockName: this.blockName,
        round: this._rounds,
        candidates,
        best: this.best(),
        measured: this.benchmarkRunner !== null,
      });
    }
    this._rounds++;
    return Math.max(0, now - prev);
  }

  best(): (ScoredCandidate & { medianMs: number | null; minMs: number | null }) | null {
    if (!this._best) return null;
    return {
      sketchName: this._best.sketchName,
      params: this._best.params,
      score: this._best.score,
      medianMs: this._best.medianMs ?? null,
      minMs: this._best.minMs ?? null
    };
  }

  bestTrace(): SerializedStep[] | null {
    if (!this._best) return null;
    const sketch = this.sketchByName.get(this._best.sketchName);
    if (!sketch) return null;
    try {
      const sch = new Schedule(clonePrimFunc(this.primFunc));
      sketch.instantiate(this._best.params)(sch, this.blockName, this.target);
      return sch.trace.serialize();
    } catch (e) {
      this._warn('best-trace', this.blockName, e);
      return null;
    }
  }

  _produceCandidates(): ScoredCandidate[] {
    if (this.enumSketch) {
      const scored: ScoredCandidate[] = [];
      for (const params of this.enumParams as SketchParams[]) {
        const r = this._evaluate(this.enumSketch, params);
        if (r) scored.push({ sketchName: this.enumSketch.name, params, score: r.score });
      }
      scored.sort((a, b) => b.score - a.score);
      return scored;
    }
    const evaluator = (sketch: ScheduleSketch, params: SketchParams) => this._evaluate(sketch, params);
    const { candidates, population } = this.strategy.search(this.sketches, evaluator, this.population);
    this.population = population;
    return candidates;
  }

  _evaluate(sketch: ScheduleSketch, params: SketchParams): { score: number } | null {
    try {
      const cloned = clonePrimFunc(this.evalFunc);
      const sch = new Schedule(cloned);
      sketch.instantiate(params)(sch, this.evalBlockName, this.target);
      const errors = ScheduleValidator.validate(cloned);
      if (errors.length > 0) return null;
      const blockLimit = this.target.maxThreadsPerBlock;
      if (this.target.isGPU && this.target.isGPU() && blockLimit && gpuThreadBlockSize(cloned) > blockLimit) return null;
      return { score: this.costModel.score(cloned) };
    } catch (e) {
      if (!this._warnedEvalSketches.has(sketch.name)) {
        this._warnedEvalSketches.add(sketch.name);
        this._warn('evaluate-candidate', this.blockName, e);
      }
      return null;
    }
  }

  _measureAndLearn(candidates: readonly ScoredCandidate[]): void {
    const topK = candidates.slice(0, this.config.topKForBenchmark);
    for (const cand of topK) {
      if (this.deadline && this.deadline.expired) break;
      const measured = this._measure(cand);
      if (!measured) continue;
      const score = -measured.result.medianMs;
      (this.learnedModel as LearnedCostModel).addSample(measured.features, score);
      this._consider({
        sketchName: cand.sketchName, params: cand.params, score,
        measuredScore: score, medianMs: measured.result.medianMs, minMs: measured.result.minMs
      });
    }
    (this.learnedModel as LearnedCostModel).train();
  }

  _measure(cand: ScoredCandidate): { result: BenchmarkResult; features: number[][] } | null {
    const sketch = this.sketchByName.get(cand.sketchName);
    if (!sketch) return null;
    let scheduled: PrimFunc, miniScheduled: PrimFunc;
    try {
      scheduled = clonePrimFunc(this.primFunc);
      sketch.instantiate(cand.params)(new Schedule(scheduled), this.blockName, this.target);
      miniScheduled = clonePrimFunc(this.evalFunc);
      sketch.instantiate(cand.params)(new Schedule(miniScheduled), this.evalBlockName, this.target);
    } catch (e) {
      this._warn('measure-candidate', this.blockName, e);
      return null;
    }
    const result = (this.benchmarkRunner as BenchmarkRunnerLike).run(scheduled);
    if (!result) return null;
    return { result, features: FeatureExtractor.extractStatements(miniScheduled) };
  }

  _consider(rec: TuningRecordDraft): void {
    if (!this._best || rec.measuredScore > this._best.measuredScore) this._best = rec;
  }
}
