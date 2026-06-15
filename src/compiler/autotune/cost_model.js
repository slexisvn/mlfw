import { FeatureExtractor } from './features.js';

class CostEstimate {
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
    this.hidden = opts.hidden ?? 8;
    this.lr = opts.lr ?? 0.05;
    this.epochs = opts.epochs ?? 2000;
    this.seed = opts.seed ?? 1;
    this._net = state || null;
    this._trainingData = [];
  }

  addSample(features, measuredScore) {
    this._trainingData.push({ features: features.toVector(), score: measuredScore });
  }

  _normalizers() {
    const dim = this._trainingData[0].features.length;
    const mean = new Array(dim).fill(0);
    for (const s of this._trainingData) for (let i = 0; i < dim; i++) mean[i] += s.features[i];
    for (let i = 0; i < dim; i++) mean[i] /= this._trainingData.length;
    const std = new Array(dim).fill(0);
    for (const s of this._trainingData) for (let i = 0; i < dim; i++) { const d = s.features[i] - mean[i]; std[i] += d * d; }
    for (let i = 0; i < dim; i++) std[i] = Math.sqrt(std[i] / this._trainingData.length) || 1;

    let yMean = 0;
    for (const s of this._trainingData) yMean += s.score;
    yMean /= this._trainingData.length;
    let yVar = 0;
    for (const s of this._trainingData) { const d = s.score - yMean; yVar += d * d; }
    const yStd = Math.sqrt(yVar / this._trainingData.length) || 1;
    return { mean, std, yMean, yStd };
  }

  train() {
    if (this._trainingData.length === 0) return;
    const dim = this._trainingData[0].features.length;
    const H = this.hidden;
    const { mean, std, yMean, yStd } = this._normalizers();

    const rng = this._rng();
    const r1 = 1 / Math.sqrt(dim);
    const r2 = 1 / Math.sqrt(H);
    const W1 = Array.from({ length: H }, () => Array.from({ length: dim }, () => (rng() * 2 - 1) * r1));
    const b1 = new Array(H).fill(0);
    const W2 = Array.from({ length: H }, () => (rng() * 2 - 1) * r2);
    let b2 = 0;

    const xs = this._trainingData.map(s => s.features.map((v, i) => (v - mean[i]) / std[i]));
    const ys = this._trainingData.map(s => (s.score - yMean) / yStd);
    const N = xs.length;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let n = 0; n < N; n++) {
        const x = xs[n];
        const z = new Array(H);
        const a = new Array(H);
        for (let h = 0; h < H; h++) {
          let acc = b1[h];
          const w = W1[h];
          for (let i = 0; i < dim; i++) acc += w[i] * x[i];
          z[h] = acc;
          a[h] = Math.tanh(acc);
        }
        let pred = b2;
        for (let h = 0; h < H; h++) pred += W2[h] * a[h];

        const dErr = pred - ys[n];
        b2 -= this.lr * dErr;
        for (let h = 0; h < H; h++) {
          const dA = dErr * W2[h];
          const dZ = dA * (1 - a[h] * a[h]);
          W2[h] -= this.lr * dErr * a[h];
          const w = W1[h];
          for (let i = 0; i < dim; i++) w[i] -= this.lr * dZ * x[i];
          b1[h] -= this.lr * dZ;
        }
      }
    }

    this._net = { dim, hidden: H, W1, b1, W2, b2, mean, std, yMean, yStd };
  }

  _rng() {
    let s = (this.seed >>> 0) || 1;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }

  predict(features) {
    if (!this._net) return 0;
    const { dim, hidden, W1, b1, W2, b2, mean, std, yMean, yStd } = this._net;
    const vec = features.toVector();
    const x = new Array(dim);
    for (let i = 0; i < dim; i++) x[i] = ((vec[i] || 0) - mean[i]) / std[i];
    let pred = b2;
    for (let h = 0; h < hidden; h++) {
      let acc = b1[h];
      const w = W1[h];
      for (let i = 0; i < dim; i++) acc += w[i] * x[i];
      pred += W2[h] * Math.tanh(acc);
    }
    return pred * yStd + yMean;
  }

  get trained() {
    return this._net !== null;
  }

  serialize() {
    return { net: this._net, numSamples: this._trainingData.length };
  }

  static deserialize(data) {
    return new LearnedCostModel(data.net || data.weights || null);
  }
}
