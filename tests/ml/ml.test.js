import { describe, it, expect } from 'vitest';
import { tensor, ml } from '../../src/index.js';
import { makeRng } from '../../src/ml/_random.js';

function blobs(seed = 1) {
  const rng = makeRng(seed);
  const rows = [];
  const lab = [];
  const centers = [[0, 0], [8, 8], [0, 8]];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < 30; i++) {
      rows.push([centers[c][0] + rng() * 1.5, centers[c][1] + rng() * 1.5]);
      lab.push(c);
    }
  }
  return { X: tensor(rows), y: tensor(lab) };
}

describe('preprocessing', () => {
  it('StandardScaler standardizes columns', () => {
    const s = new ml.StandardScaler().fit(tensor([[1, 10], [3, 20], [5, 30]]));
    expect(s.mean_[0]).toBeCloseTo(3);
    expect(s.mean_[1]).toBeCloseTo(20);
    const Z = s.transform(tensor([[1, 10], [3, 20], [5, 30]])).toArray();
    let mean = 0;
    for (const r of Z) mean += r[0];
    expect(mean / 3).toBeCloseTo(0);
  });
});

describe('metrics', () => {
  it('r2 and accuracy', () => {
    expect(ml.r2_score(tensor([1, 2, 3]), tensor([1, 2, 3]))).toBeCloseTo(1);
    expect(ml.accuracy_score(tensor([0, 1, 1]), tensor([0, 1, 0]))).toBeCloseTo(2 / 3);
  });
});

describe('linear_model', () => {
  it('LinearRegression recovers coefficients', () => {
    const X = tensor([[1, 1], [2, 1], [1, 2], [3, 2], [2, 3], [4, 4]]);
    const y = tensor([6, 8, 9, 13, 14, 21]);
    const lr = new ml.LinearRegression().fit(X, y);
    expect(lr.score(X, y)).toBeGreaterThan(0.999);
  });
  it('Ridge shrinks but fits well', () => {
    const X = tensor([[1, 1], [2, 1], [1, 2], [3, 2], [2, 3], [4, 4]]);
    const y = tensor([6, 8, 9, 13, 14, 21]);
    expect(new ml.Ridge({ alpha: 0.01 }).fit(X, y).score(X, y)).toBeGreaterThan(0.99);
  });
  it('Lasso zeros an irrelevant feature', () => {
    const X = tensor([[1, 0.3], [2, -0.1], [3, 0.2], [4, 0.5], [5, -0.4]]);
    const y = tensor([2, 4, 6, 8, 10]);
    const la = new ml.Lasso({ alpha: 0.1 }).fit(X, y);
    const coef = la.coef_.toArray();
    expect(Math.abs(coef[1])).toBeLessThan(Math.abs(coef[0]));
    expect(la.score(X, y)).toBeGreaterThan(0.98);
  });
  it('LogisticRegression separable', () => {
    const { X, y } = blobs();
    expect(new ml.LogisticRegression({ maxIter: 300 }).fit(X, y).score(X, y)).toBeGreaterThan(0.95);
  });
});

describe('decomposition.PCA', () => {
  it('recovers principal axis and variance ratio sums to 1 at full rank', () => {
    const X = tensor([[1, 1.1], [2, 1.9], [3, 3.2], [4, 3.8], [5, 5.1]]);
    const pca = new ml.PCA().fit(X);
    const total = pca.explainedVarianceRatio_.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1, 4);
    expect(pca.explainedVarianceRatio_[0]).toBeGreaterThan(0.98);
    const Z = new ml.PCA({ nComponents: 1 }).fit_transform(X);
    expect(Z.shape[1]).toBe(1);
  });
});

describe('cluster.KMeans', () => {
  it('recovers 3 blobs (pure clusters)', () => {
    const { X, y } = blobs();
    const km = new ml.KMeans({ nClusters: 3, randomState: 0 }).fit(X);
    const labels = km.labels_.toArray();
    const yy = y.toArray();
    const groups = new Map();
    for (let i = 0; i < labels.length; i++) {
      const key = labels[i];
      if (!groups.has(key)) groups.set(key, new Set());
      groups.get(key).add(yy[i]);
    }
    for (const s of groups.values()) expect(s.size).toBe(1);
    expect(km.inertia_).toBeGreaterThan(0);
  });
});

describe('neighbors + naive_bayes', () => {
  it('KNN and GaussianNB high accuracy', () => {
    const { X, y } = blobs();
    expect(new ml.KNeighborsClassifier({ nNeighbors: 5 }).fit(X, y).score(X, y)).toBeGreaterThan(0.95);
    expect(new ml.GaussianNB().fit(X, y).score(X, y)).toBeGreaterThan(0.95);
  });
  it('KNN regressor', () => {
    const X = tensor([[0], [1], [2], [10], [11], [12]]);
    const y = tensor([0, 0, 0, 10, 10, 10]);
    const r = new ml.KNeighborsRegressor({ nNeighbors: 3 }).fit(X, y);
    expect(r.predict(tensor([[1]])).toArray()[0]).toBeCloseTo(0);
  });
});

describe('trees', () => {
  it('DecisionTree fits XOR (classifier)', () => {
    const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const y = tensor([0, 1, 1, 0]);
    expect(new ml.DecisionTreeClassifier().fit(X, y).score(X, y)).toBe(1);
  });
  it('RandomForest and GradientBoosting classify blobs', () => {
    const { X, y } = blobs();
    expect(new ml.RandomForestClassifier({ nEstimators: 15, randomState: 0 }).fit(X, y).score(X, y)).toBeGreaterThan(0.95);
    expect(new ml.GradientBoostingClassifier({ nEstimators: 25, randomState: 0 }).fit(X, y).score(X, y)).toBeGreaterThan(0.95);
  });
  it('GradientBoostingRegressor reduces error', () => {
    const rng = makeRng(3);
    const rows = []; const ys = [];
    for (let i = 0; i < 60; i++) { const a = rng() * 4; const b = rng() * 4; rows.push([a, b]); ys.push(a * a - b + rng() * 0.1); }
    const X = tensor(rows); const y = tensor(ys);
    const gb = new ml.GradientBoostingRegressor({ nEstimators: 50, maxDepth: 3, randomState: 0 }).fit(X, y);
    expect(gb.score(X, y)).toBeGreaterThan(0.9);
  });
});

describe('model_selection', () => {
  it('train_test_split sizes and determinism', () => {
    const { X, y } = blobs();
    const [a, b] = ml.train_test_split(X, y, { testSize: 0.3, randomState: 0 });
    expect(a.shape[0] + b.shape[0]).toBe(90);
    expect(b.shape[0]).toBe(27);
  });
  it('cross_val_score and GridSearchCV', () => {
    const { X, y } = blobs();
    const scores = ml.cross_val_score(() => new ml.GaussianNB(), X, y, { cv: 3 });
    expect(scores.length).toBe(3);
    expect(scores.reduce((p, q) => p + q, 0) / 3).toBeGreaterThan(0.9);
    const gs = new ml.GridSearchCV((p) => new ml.DecisionTreeClassifier(p), { maxDepth: [1, 3, 5] }, { cv: 3 }).fit(X, y);
    expect(gs.bestScore_).toBeGreaterThan(0.9);
  });
  it('PurgedKFold embargo removes neighbors', () => {
    const folds = new ml.PurgedKFold({ nSplits: 5, embargo: 2 }).split(50);
    for (const { train, test } of folds) {
      const lo = test[0]; const hi = test[test.length - 1];
      for (const t of train) expect(t < lo - 2 || t > hi + 2).toBe(true);
    }
  });
});
