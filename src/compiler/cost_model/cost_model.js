import { FeatureExtractor } from './features.js';

export class CostEstimate {
  constructor(score, breakdown) {
    this.score = score;
    this.breakdown = breakdown;
  }
}

export class AnalyticalCostModel {
  constructor(target) {
    this.target = target;
    this._weights = {
      parallelism: 2.0,
      vectorization: 1.5,
      memoryCoalescing: 2.0,
      occupancy: 1.0,
      arithmeticIntensity: 1.0,
      loopOverhead: -0.5,
      codeSize: -0.3
    };
  }

  estimate(primFunc) {
    const features = FeatureExtractor.extract(primFunc);
    return this.estimateFromFeatures(features);
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
    return Math.min(1.0, f.arithmeticIntensity * 10);
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
  constructor(weights = null) {
    this._weights = weights || null;
    this._trainingData = [];
  }

  addSample(features, measuredScore) {
    this._trainingData.push({ features: features.toVector(), score: measuredScore });
  }

  train() {
    if (this._trainingData.length === 0) return;
    const dim = this._trainingData[0].features.length;
    this._weights = new Array(dim).fill(0);

    const lr = 0.001;
    const epochs = 100;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const sample of this._trainingData) {
        let predicted = 0;
        for (let i = 0; i < dim; i++) {
          predicted += this._weights[i] * sample.features[i];
        }
        const error = sample.score - predicted;
        for (let i = 0; i < dim; i++) {
          this._weights[i] += lr * error * sample.features[i];
        }
      }
    }
  }

  predict(features) {
    if (!this._weights) return 0;
    const vec = features.toVector();
    let score = 0;
    for (let i = 0; i < vec.length; i++) {
      score += this._weights[i] * (vec[i] || 0);
    }
    return score;
  }

  get trained() {
    return this._weights !== null && this._trainingData.length > 0;
  }

  serialize() {
    return { weights: this._weights, numSamples: this._trainingData.length };
  }

  static deserialize(data) {
    const model = new LearnedCostModel(data.weights);
    return model;
  }
}
