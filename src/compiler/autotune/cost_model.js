import { FeatureExtractor } from './features.js';
import { GradientBoostedTrees } from './gbt.js';

function aggregateStatements(stmtVecs) {
  const dim = stmtVecs[0].length;
  const out = new Array(dim + 1).fill(0);
  for (const v of stmtVecs) for (let i = 0; i < dim; i++) out[i] += v[i] || 0;
  out[dim] = stmtVecs.length;
  return out;
}

class CostEstimate {
  constructor(score, breakdown) {
    this.score = score;
    this.breakdown = breakdown;
  }
}

const DEFAULT_COST_WEIGHTS = {
  parallelism: 2.0,
  vectorization: 1.5,
  memoryCoalescing: 2.0,
  occupancy: 1.0,
  arithmeticIntensity: 1.0,
  loopOverhead: -0.5,
  codeSize: -0.3
};

export class AnalyticalCostModel {
  constructor(target, opts = {}) {
    this.target = target;
    this._weights = {
      ...DEFAULT_COST_WEIGHTS,
      ...(target && target.costModelWeights ? target.costModelWeights : {}),
      ...(opts.weights || {})
    };
  }

  estimate(primFunc) {
    const features = FeatureExtractor.extract(primFunc);
    return this.estimateFromFeatures(features);
  }

  score(primFunc) {
    return this.estimate(primFunc).score;
  }

  estimateFromFeatures(features) {
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

  _scoreParallelism(f) {
    if (this.target.isGPU()) {
      const totalThreads = f.threadBlockSize * f.gridSize;
      const maxPar = this.target.maxParallelism();
      return Math.min(1.0, totalThreads / Math.max(maxPar * 0.1, 1));
    }
    const ratio = f.numParallelLoops / Math.max(f.numLoops, 1);
    return ratio;
  }

  _scoreVectorization(f) {
    if (f.numLoops === 0) return 0;
    if (this.target.isGPU()) {
      const coalescedRatio = f.strideOneAccesses / Math.max(f.strideOneAccesses + f.nonStrideOneAccesses, 1);
      return coalescedRatio;
    }
    return f.numVectorizedLoops > 0 ? Math.min(1.0, f.innermostExtent / this.target.vectorWidth) : 0;
  }

  _scoreMemoryAccess(f) {
    const total = f.strideOneAccesses + f.nonStrideOneAccesses;
    if (total === 0) return 1.0;
    return f.strideOneAccesses / total;
  }

  _scoreOccupancy(f) {
    if (!this.target.isGPU()) return 1.0;
    if (f.threadBlockSize === 0) return 0;
    const warpSize = this.target.warpSize;
    const warpsPerBlock = Math.ceil(f.threadBlockSize / warpSize);
    const maxWarps = Math.floor(this.target.maxThreadsPerBlock / warpSize);
    return Math.min(1.0, warpsPerBlock / maxWarps);
  }

  _scoreIntensity(f) {
    const boost = this.target.supportsFloat16 ? 1.5 : 1.0;
    return Math.min(1.0, f.arithmeticIntensity * 10 * boost);
  }

  _scoreOverhead(f) {
    return f.numSerialLoops / Math.max(f.numLoops, 1);
  }

  _scoreCodeSize(f) {
    return Math.min(1.0, (f.numMathOps + f.numExternCalls) / 256);
  }

  compare(primFuncA, primFuncB) {
    return this.estimate(primFuncA).score - this.estimate(primFuncB).score;
  }
}

export class LearnedCostModel {
  constructor(state = null, opts = {}) {
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

  addSample(stmtVecs, measuredScore) {
    if (!stmtVecs || stmtVecs.length === 0) return;
    if (!Number.isFinite(measuredScore)) return;
    this._X.push(aggregateStatements(stmtVecs));
    this._Y.push(measuredScore);
  }

  train() {
    if (this._X.length === 0) return;
    const gbt = new GradientBoostedTrees(this.opts);
    gbt.fit(this._X, this._Y);
    this._gbt = gbt;
  }

  predict(stmtVecs) {
    if (!this._gbt || !stmtVecs || stmtVecs.length === 0) return 0;
    return this._gbt.predict(aggregateStatements(stmtVecs));
  }

  get trained() {
    return this._gbt !== null;
  }

  serialize() {
    return { gbt: this._gbt ? this._gbt.serialize() : null, numSamples: this._X.length };
  }

  static deserialize(data) {
    return new LearnedCostModel(data && data.gbt ? data.gbt : null);
  }
}

export class GuidedCostModel {
  constructor(analytical, learned) {
    this.analytical = analytical;
    this.learned = learned;
  }

  score(primFunc) {
    if (this.learned && this.learned.trained) {
      return this.learned.predict(FeatureExtractor.extractStatements(primFunc));
    }
    return this.analytical.score(primFunc);
  }
}
