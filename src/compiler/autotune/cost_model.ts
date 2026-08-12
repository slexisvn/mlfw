import { FeatureExtractor, STATEMENT_FEATURE_SCHEMA } from './features.js';
import type { ScheduleFeatureSet } from './features.js';
import { GradientBoostedTrees } from './gbt.js';
import type { SerializedGBT } from './gbt.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

export type CostWeightMap = Record<string, number>;
export type StatementVectors = readonly (readonly number[])[];
export type ScheduleFeatureView = ScheduleFeatureSet;
export type SerializedLearnedModel = { gbt: SerializedGBT | null; numSamples: number };

export type CostModelTarget = ScheduleTarget & { costModelWeights?: CostWeightMap; maxParallelism?: () => number; supportsFloat16?: boolean };

const MAX_FEATURE_NAMES = new Set(['depth', 'threadBlockSize', 'gridSize', 'underReduction', 'vectorized', 'parallelized', 'innermostExtent']);
const MEAN_FEATURE_NAMES = new Set(['arithmeticIntensity']);
const MAX_FEATURE_IDX = new Set(STATEMENT_FEATURE_SCHEMA.map((n, i) => (MAX_FEATURE_NAMES.has(n) ? i : -1)).filter(i => i >= 0));
const MEAN_FEATURE_IDX = new Set(STATEMENT_FEATURE_SCHEMA.map((n, i) => (MEAN_FEATURE_NAMES.has(n) ? i : -1)).filter(i => i >= 0));

function aggregateStatements(stmtVecs: StatementVectors): number[] {
  const dim = stmtVecs[0].length;
  const out: number[] = new Array<number>(dim + 1).fill(0);
  for (const v of stmtVecs) {
    for (let i = 0; i < dim; i++) {
      const x = v[i] || 0;
      if (MAX_FEATURE_IDX.has(i)) { if (x > out[i]) out[i] = x; }
      else out[i] += x;
    }
  }
  for (const i of MEAN_FEATURE_IDX) if (i < dim) out[i] /= stmtVecs.length;
  out[dim] = stmtVecs.length;
  return out;
}

class CostEstimate {
  score: number;
  breakdown: CostWeightMap;

  constructor(score: number, breakdown: CostWeightMap) {
    this.score = score;
    this.breakdown = breakdown;
  }
}

const DEFAULT_COST_WEIGHTS: CostWeightMap = {
  parallelism: 2.0,
  vectorization: 1.5,
  memoryCoalescing: 2.0,
  occupancy: 1.0,
  arithmeticIntensity: 1.0,
  loopOverhead: -0.5,
  codeSize: -0.3
};

export class AnalyticalCostModel {
  target: CostModelTarget;
  private _weights: CostWeightMap;

  constructor(target: CostModelTarget, opts: Readonly<{ weights?: CostWeightMap }> = {}) {
    this.target = target;
    this._weights = {
      ...DEFAULT_COST_WEIGHTS,
      ...(target && target.costModelWeights ? target.costModelWeights : {}),
      ...(opts.weights || {})
    };
  }

  estimate(primFunc: PrimFunc): CostEstimate {
    const features = FeatureExtractor.extract(primFunc);
    return this.estimateFromFeatures(features);
  }

  score(primFunc: PrimFunc): number {
    return this.estimate(primFunc).score;
  }

  estimateFromFeatures(features: ScheduleFeatureView): CostEstimate {
    const parallelismScore = this._scoreParallelism(features);
    const vectorScore = this._scoreVectorization(features);
    const memoryScore = this._scoreMemoryAccess(features);
    const occupancyScore = this._scoreOccupancy(features);
    const intensityScore = this._scoreIntensity(features);
    const overheadPenalty = this._scoreOverhead(features);
    const codeSizePenalty = this._scoreCodeSize(features);

    const w = this._weights;
    const score =
      w.parallelism * parallelismScore +
      w.vectorization * vectorScore +
      w.memoryCoalescing * memoryScore +
      w.occupancy * occupancyScore +
      w.arithmeticIntensity * intensityScore +
      w.loopOverhead * overheadPenalty +
      w.codeSize * codeSizePenalty;

    return new CostEstimate(score, {
      parallelism: parallelismScore,
      vectorization: vectorScore,
      memoryCoalescing: memoryScore,
      occupancy: occupancyScore,
      arithmeticIntensity: intensityScore,
      loopOverhead: overheadPenalty,
      codeSize: codeSizePenalty
    });
  }

  _scoreParallelism(f: ScheduleFeatureView): number {
    if (this.target.isGPU()) {
      const totalThreads = f.threadBlockSize * f.gridSize;
      const maxPar = (this.target.maxParallelism as () => number)();
      return Math.min(1.0, totalThreads / Math.max(maxPar * 0.1, 1));
    }
    const ratio = f.numParallelLoops / Math.max(f.numLoops, 1);
    return ratio;
  }

  _scoreVectorization(f: ScheduleFeatureView): number {
    if (f.numLoops === 0) return 0;
    if (this.target.isGPU()) {
      const coalescedRatio = f.strideOneAccesses / Math.max(f.strideOneAccesses + f.nonStrideOneAccesses, 1);
      return coalescedRatio;
    }
    return f.numVectorizedLoops > 0 ? Math.min(1.0, f.innermostExtent / this.target.vectorWidth) : 0;
  }

  _scoreMemoryAccess(f: ScheduleFeatureView): number {
    const total = f.strideOneAccesses + f.nonStrideOneAccesses;
    if (total === 0) return 1.0;
    return f.strideOneAccesses / total;
  }

  _scoreOccupancy(f: ScheduleFeatureView): number {
    if (!this.target.isGPU()) return 1.0;
    if (f.threadBlockSize === 0) return 0;
    const warpSize = this.target.warpSize as number;
    const warpsPerBlock = Math.ceil(f.threadBlockSize / warpSize);
    const maxWarps = Math.floor(this.target.maxThreadsPerBlock / warpSize);
    return Math.min(1.0, warpsPerBlock / maxWarps);
  }

  _scoreIntensity(f: ScheduleFeatureView): number {
    const boost = this.target.supportsFloat16 ? 1.5 : 1.0;
    return Math.min(1.0, f.arithmeticIntensity * 10 * boost);
  }

  _scoreOverhead(f: ScheduleFeatureView): number {
    return f.numSerialLoops / Math.max(f.numLoops, 1);
  }

  _scoreCodeSize(f: ScheduleFeatureView): number {
    return Math.min(1.0, (f.numMathOps + f.numExternCalls) / 256);
  }

  compare(primFuncA: PrimFunc, primFuncB: PrimFunc): number {
    return this.estimate(primFuncA).score - this.estimate(primFuncB).score;
  }
}

export class LearnedCostModel {
  opts: { numTrees: number; maxDepth: number; lr: number; minSamples: number };
  private _gbt: GradientBoostedTrees | null;
  private _X: number[][];
  private _Y: number[];

  constructor(state: SerializedGBT | null = null, opts: Readonly<{ numTrees?: number; maxDepth?: number; lr?: number; minSamples?: number }> = {}) {
    this.opts = {
      numTrees: opts.numTrees ?? 60,
      maxDepth: opts.maxDepth ?? 3,
      lr: opts.lr ?? 0.1,
      minSamples: opts.minSamples ?? 1
    };
    this._gbt = state ? GradientBoostedTrees.deserialize(state) : null;
    this._X = [];
    this._Y = [];
  }

  addSample(stmtVecs: StatementVectors | null | undefined, measuredScore: number): void {
    if (!stmtVecs || stmtVecs.length === 0) return;
    if (!Number.isFinite(measuredScore)) return;
    this._X.push(aggregateStatements(stmtVecs));
    this._Y.push(measuredScore);
  }

  train(): void {
    if (this._X.length === 0) return;
    const gbt = new GradientBoostedTrees(this.opts);
    gbt.fit(this._X, this._Y);
    this._gbt = gbt;
  }

  predict(stmtVecs: StatementVectors | null | undefined): number {
    if (!this._gbt || !stmtVecs || stmtVecs.length === 0) return 0;
    return this._gbt.predict(aggregateStatements(stmtVecs));
  }

  get trained(): boolean {
    return this._gbt !== null;
  }

  get sampleCount(): number {
    return this._X.length;
  }

  serialize(): SerializedLearnedModel {
    return { gbt: this._gbt ? this._gbt.serialize() : null, numSamples: this._X.length };
  }

  static deserialize(data: SerializedLearnedModel | null): LearnedCostModel {
    return new LearnedCostModel(data && data.gbt ? data.gbt : null);
  }
}

export class GuidedCostModel {
  analytical: AnalyticalCostModel;
  learned: LearnedCostModel | null;
  confidenceSamples: number;

  constructor(analytical: AnalyticalCostModel, learned: LearnedCostModel | null, opts: Readonly<{ confidenceSamples?: number }> = {}) {
    this.analytical = analytical;
    this.learned = learned;
    this.confidenceSamples = opts.confidenceSamples ?? 8;
  }

  _learnedConfident(): boolean {
    return !!(this.learned && this.learned.trained &&
           this.learned.sampleCount >= this.confidenceSamples);
  }

  score(primFunc: PrimFunc): number {
    if (this._learnedConfident()) {
      return (this.learned as LearnedCostModel).predict(FeatureExtractor.extractStatements(primFunc));
    }
    return this.analytical.score(primFunc);
  }
}
