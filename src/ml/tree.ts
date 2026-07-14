import { _dispatch } from '../tensor/ops/ops.js';
import { tensor } from '../tensor/factory/from_ops.js';
import { r2_score, accuracy_score } from './metrics.js';
import { vectorOf, encodeLabels, takeRows } from './_util.js';
import { makeRng } from './_random.js';
import type { MLTensor } from './types.js';

const NO_LIMIT = 1 << 30;

type TreeParams = {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: number;
  randomState?: number;
};
type ResolvedTreeParams = Required<Pick<TreeParams, 'maxDepth' | 'minSamplesSplit' | 'minSamplesLeaf' | 'maxFeatures'>>;
type TreeNodes = [MLTensor, MLTensor, MLTensor, MLTensor, MLTensor];
type Stage = TreeNodes[];

function fitTree(X: MLTensor, y: MLTensor, params: ResolvedTreeParams, classify: boolean, seed: number): TreeNodes {
  return _dispatch('decision_tree_fit', X, y,
    params.maxDepth, params.minSamplesSplit, params.minSamplesLeaf, params.maxFeatures, classify, seed) as TreeNodes;
}

function treePredict(X: MLTensor, nodes: TreeNodes): MLTensor {
  return _dispatch('decision_tree_predict', X, nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]) as MLTensor;
}

class BaseTree {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures: number;
  randomState: number;
  protected _classify: boolean;
  protected _nodes: TreeNodes | null;

  constructor(params: TreeParams, classify: boolean) {
    this.maxDepth = params.maxDepth ?? NO_LIMIT;
    this.minSamplesSplit = params.minSamplesSplit ?? 2;
    this.minSamplesLeaf = params.minSamplesLeaf ?? 1;
    this.maxFeatures = params.maxFeatures ?? 0;
    this.randomState = params.randomState ?? 0;
    this._classify = classify;
    this._nodes = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    this._nodes = fitTree(X, y, this, this._classify, this.randomState);
    return this;
  }

  predict(X: MLTensor): MLTensor {
    return treePredict(X, this._nodes!);
  }
}

export class DecisionTreeRegressor extends BaseTree {
  constructor(params: TreeParams = {}) { super(params, false); }
  score(X: MLTensor, y: MLTensor): number { return r2_score(y, this.predict(X)); }
}

export class DecisionTreeClassifier extends BaseTree {
  constructor(params: TreeParams = {}) { super(params, true); }
  score(X: MLTensor, y: MLTensor): number { return accuracy_score(y, this.predict(X)); }
}

function defaultMaxFeatures(mf: number, d: number, classify: boolean): number {
  if (mf > 0) return mf;
  return classify ? Math.max(1, Math.floor(Math.sqrt(d))) : Math.max(1, Math.floor(d / 3));
}

class BaseForest {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures: number;
  randomState: number;
  protected _classify: boolean;
  protected _trees: TreeNodes[];

  constructor(params: TreeParams & { nEstimators?: number }, classify: boolean) {
    this.nEstimators = params.nEstimators ?? 100;
    this.maxDepth = params.maxDepth ?? NO_LIMIT;
    this.minSamplesSplit = params.minSamplesSplit ?? 2;
    this.minSamplesLeaf = params.minSamplesLeaf ?? 1;
    this.maxFeatures = params.maxFeatures ?? 0;
    this.randomState = params.randomState ?? 0;
    this._classify = classify;
    this._trees = [];
  }

  fit(X: MLTensor, y: MLTensor): this {
    const n = X.shape[0];
    const d = X.shape[1];
    const rng = makeRng(this.randomState);
    const mf = defaultMaxFeatures(this.maxFeatures, d, this._classify);
    const params: ResolvedTreeParams = { maxDepth: this.maxDepth, minSamplesSplit: this.minSamplesSplit, minSamplesLeaf: this.minSamplesLeaf, maxFeatures: mf };
    this._trees = [];
    for (let e = 0; e < this.nEstimators; e++) {
      const idx = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = Math.floor(rng() * n);
      const Xb = takeRows(X, idx);
      const yb = takeRows(y, idx);
      this._trees.push(fitTree(Xb, yb, params, this._classify, this.randomState + e + 1));
    }
    return this;
  }

  predict(X: MLTensor): MLTensor {
    const n = X.shape[0];
    const preds = this._trees.map((t: TreeNodes) => treePredict(X, t).toArray() as ArrayLike<number>);
    const out = new Float64Array(n);
    if (this._classify) {
      for (let i = 0; i < n; i++) {
        const votes = new Map();
        let bestLabel = preds[0][i];
        let bestCount = 0;
        for (let e = 0; e < preds.length; e++) {
          const label = preds[e][i];
          const c = (votes.get(label) || 0) + 1;
          votes.set(label, c);
          if (c > bestCount) { bestCount = c; bestLabel = label; }
        }
        out[i] = bestLabel;
      }
    } else {
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let e = 0; e < preds.length; e++) s += preds[e][i];
        out[i] = s / preds.length;
      }
    }
    return tensor(out, { shape: [n], dtype: X.dtype, device: X.device }) as MLTensor;
  }
}

export class RandomForestRegressor extends BaseForest {
  constructor(params: TreeParams & { nEstimators?: number } = {}) { super(params, false); }
  score(X: MLTensor, y: MLTensor): number { return r2_score(y, this.predict(X)); }
}

export class RandomForestClassifier extends BaseForest {
  constructor(params: TreeParams & { nEstimators?: number } = {}) { super(params, true); }
  score(X: MLTensor, y: MLTensor): number { return accuracy_score(y, this.predict(X)); }
}

export class GradientBoostingRegressor {
  nEstimators: number;
  learningRate: number;
  params: ResolvedTreeParams;
  randomState: number;
  init_: number;
  private _trees: TreeNodes[];

  constructor({ nEstimators = 100, learningRate = 0.1, maxDepth = 3, minSamplesSplit = 2, minSamplesLeaf = 1, randomState = 0 }: TreeParams & { nEstimators?: number; learningRate?: number } = {}) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.params = { maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures: 0 };
    this.randomState = randomState;
    this.init_ = 0;
    this._trees = [];
  }

  fit(X: MLTensor, y: MLTensor): this {
    const yv = vectorOf(y);
    const n = yv.n;
    let init = 0;
    for (let i = 0; i < n; i++) init += yv.data[i];
    init /= n;
    this.init_ = init;
    const F = new Float64Array(n).fill(init);
    this._trees = [];
    for (let m = 0; m < this.nEstimators; m++) {
      const residual = new Float64Array(n);
      for (let i = 0; i < n; i++) residual[i] = yv.data[i] - F[i];
      const rT = tensor(residual, { shape: [n], dtype: X.dtype, device: X.device }) as MLTensor;
      const nodes = fitTree(X, rT, this.params, false, this.randomState + m + 1);
      this._trees.push(nodes);
      const pred = treePredict(X, nodes).toArray() as ArrayLike<number>;
      for (let i = 0; i < n; i++) F[i] += this.learningRate * pred[i];
    }
    return this;
  }

  predict(X: MLTensor): MLTensor {
    const n = X.shape[0];
    const F = new Float64Array(n).fill(this.init_);
    for (const nodes of this._trees) {
      const pred = treePredict(X, nodes).toArray() as ArrayLike<number>;
      for (let i = 0; i < n; i++) F[i] += this.learningRate * pred[i];
    }
    return tensor(F, { shape: [n], dtype: X.dtype, device: X.device }) as MLTensor;
  }

  score(X: MLTensor, y: MLTensor): number {
    return r2_score(y, this.predict(X));
  }
}

export class GradientBoostingClassifier {
  nEstimators: number;
  learningRate: number;
  params: ResolvedTreeParams;
  randomState: number;
  classes_: number[] | null;
  private _stages: Stage[];

  constructor({ nEstimators = 100, learningRate = 0.1, maxDepth = 3, minSamplesSplit = 2, minSamplesLeaf = 1, randomState = 0 }: TreeParams & { nEstimators?: number; learningRate?: number } = {}) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.params = { maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures: 0 };
    this.randomState = randomState;
    this.classes_ = null;
    this._stages = [];
  }

  fit(X: MLTensor, y: MLTensor): this {
    const yv = vectorOf(y);
    const n = yv.n;
    const { y: labels, classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    const K = classes.length;
    const F = new Float64Array(n * K);
    this._stages = [];
    for (let m = 0; m < this.nEstimators; m++) {
      const P = softmaxRows(F, n, K);
      const stage: Stage = [];
      for (let c = 0; c < K; c++) {
        const grad = new Float64Array(n);
        for (let i = 0; i < n; i++) grad[i] = (labels[i] === c ? 1 : 0) - P[i * K + c];
        const gT = tensor(grad, { shape: [n], dtype: X.dtype, device: X.device }) as MLTensor;
        const nodes = fitTree(X, gT, this.params, false, this.randomState + m * K + c + 1);
        stage.push(nodes);
        const pred = treePredict(X, nodes).toArray() as ArrayLike<number>;
        for (let i = 0; i < n; i++) F[i * K + c] += this.learningRate * pred[i];
      }
      this._stages.push(stage);
    }
    return this;
  }

  predict(X: MLTensor): MLTensor {
    const n = X.shape[0];
    const classes = this.classes_!;
    const K = classes.length;
    const F = new Float64Array(n * K);
    for (const stage of this._stages) {
      for (let c = 0; c < K; c++) {
        const pred = treePredict(X, stage[c]).toArray() as ArrayLike<number>;
        for (let i = 0; i < n; i++) F[i * K + c] += this.learningRate * pred[i];
      }
    }
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let c = 1; c < K; c++) if (F[i * K + c] > F[i * K + best]) best = c;
      out[i] = classes[best];
    }
    return tensor(out, { shape: [n], dtype: X.dtype, device: X.device }) as MLTensor;
  }

  score(X: MLTensor, y: MLTensor): number {
    return accuracy_score(y, this.predict(X));
  }
}

function softmaxRows(F: Float64Array, n: number, K: number): Float64Array {
  const P = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let c = 0; c < K; c++) if (F[i * K + c] > mx) mx = F[i * K + c];
    let sum = 0;
    for (let c = 0; c < K; c++) {
      const e = Math.exp(F[i * K + c] - mx);
      P[i * K + c] = e;
      sum += e;
    }
    for (let c = 0; c < K; c++) P[i * K + c] /= sum;
  }
  return P;
}
