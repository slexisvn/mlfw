import { takeRows } from './_util.js';
import { makeRng, shuffledIndices } from './_random.js';
import type { FitPredictEstimator, MLTensor } from './types.js';

type SplitOptions = { testSize?: number; shuffle?: boolean; randomState?: number };
type Fold = { train: number[]; test: number[] };
type ScoringFn = (yTrue: MLTensor, yPred: MLTensor) => number;
type EstimatorFactory<T extends FitPredictEstimator = FitPredictEstimator> = (params?: Record<string, unknown> | null) => T;

export function train_test_split(X: MLTensor, y: MLTensor, { testSize = 0.25, shuffle = true, randomState = 0 }: SplitOptions = {}): [MLTensor, MLTensor, MLTensor, MLTensor] {
  const n = X.shape[0];
  const idx = shuffle ? shuffledIndices(n, makeRng(randomState)) : Array.from({ length: n }, (_, i) => i);
  const nTest = Math.max(1, Math.round(n * testSize));
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);
  return [takeRows(X, trainIdx), takeRows(X, testIdx), takeRows(y, trainIdx), takeRows(y, testIdx)];
}

export class KFold {
  nSplits: number;
  shuffle: boolean;
  randomState: number;

  constructor({ nSplits = 5, shuffle = false, randomState = 0 }: { nSplits?: number; shuffle?: boolean; randomState?: number } = {}) {
    this.nSplits = nSplits;
    this.shuffle = shuffle;
    this.randomState = randomState;
  }

  split(n: number): Fold[] {
    const idx = this.shuffle ? shuffledIndices(n, makeRng(this.randomState)) : Array.from({ length: n }, (_, i) => i);
    const folds: Fold[] = [];
    const base = Math.floor(n / this.nSplits);
    let rem = n % this.nSplits;
    let start = 0;
    for (let k = 0; k < this.nSplits; k++) {
      const size = base + (rem > 0 ? 1 : 0);
      if (rem > 0) rem--;
      const test = idx.slice(start, start + size);
      const train = idx.slice(0, start).concat(idx.slice(start + size));
      folds.push({ train, test });
      start += size;
    }
    return folds;
  }
}

export class TimeSeriesSplit {
  nSplits: number;

  constructor({ nSplits = 5 }: { nSplits?: number } = {}) {
    this.nSplits = nSplits;
  }

  split(n: number): Fold[] {
    const size = Math.floor(n / (this.nSplits + 1));
    const folds: Fold[] = [];
    for (let k = 1; k <= this.nSplits; k++) {
      const trainEnd = size * k;
      const testEnd = k === this.nSplits ? n : size * (k + 1);
      const train = Array.from({ length: trainEnd }, (_, i) => i);
      const test = Array.from({ length: testEnd - trainEnd }, (_, i) => trainEnd + i);
      folds.push({ train, test });
    }
    return folds;
  }
}

export function cross_val_score(makeEstimator: EstimatorFactory, X: MLTensor, y: MLTensor, { cv = 5, scoring = null, shuffle = true, randomState = 0 }: { cv?: number; scoring?: ScoringFn | null; shuffle?: boolean; randomState?: number } = {}): number[] {
  const folds = new KFold({ nSplits: cv, shuffle, randomState }).split(X.shape[0]);
  const scores = [];
  for (const { train, test } of folds) {
    const Xtr = takeRows(X, train);
    const ytr = takeRows(y, train);
    const Xte = takeRows(X, test);
    const yte = takeRows(y, test);
    const est = makeEstimator().fit(Xtr, ytr);
    scores.push(scoring ? scoring(yte, est.predict(Xte)) : est.score!(Xte, yte));
  }
  return scores;
}

function cartesian(grid: Record<string, readonly unknown[]>): Array<Record<string, unknown>> {
  const keys = Object.keys(grid);
  let combos: Array<Record<string, unknown>> = [{}];
  for (const key of keys) {
    const next: Array<Record<string, unknown>> = [];
    for (const combo of combos) {
      for (const value of grid[key]) next.push({ ...combo, [key]: value });
    }
    combos = next;
  }
  return combos;
}

export class GridSearchCV {
  makeEstimator: EstimatorFactory;
  paramGrid: Record<string, readonly unknown[]>;
  cv: number;
  scoring: ScoringFn | null;
  bestParams_: Record<string, unknown> | null;
  bestScore_: number;
  bestEstimator_: FitPredictEstimator | null;

  constructor(makeEstimator: EstimatorFactory, paramGrid: Record<string, readonly unknown[]>, { cv = 5, scoring = null }: { cv?: number; scoring?: ScoringFn | null } = {}) {
    this.makeEstimator = makeEstimator;
    this.paramGrid = paramGrid;
    this.cv = cv;
    this.scoring = scoring;
    this.bestParams_ = null;
    this.bestScore_ = -Infinity;
    this.bestEstimator_ = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    for (const params of cartesian(this.paramGrid)) {
      const scores = cross_val_score(() => this.makeEstimator(params), X, y, { cv: this.cv, scoring: this.scoring });
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
      if (mean > this.bestScore_) {
        this.bestScore_ = mean;
        this.bestParams_ = params;
      }
    }
    this.bestEstimator_ = this.makeEstimator(this.bestParams_).fit(X, y);
    return this;
  }

  predict(X: MLTensor): MLTensor {
    return this.bestEstimator_!.predict(X);
  }
}
