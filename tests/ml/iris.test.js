import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { tensor, ml } from '../../src/index.js';

function loadIris() {
  const text = readFileSync(new URL('../../examples/iris.csv', import.meta.url), 'utf8');
  const lines = text.trim().split(/\r?\n/).slice(1);
  const rows = [];
  const labels = [];
  const classMap = new Map();
  for (const line of lines) {
    const parts = line.split(',');
    if (parts.length < 5) continue;
    rows.push(parts.slice(0, 4).map(Number));
    const sp = parts[4];
    if (!classMap.has(sp)) classMap.set(sp, classMap.size);
    labels.push(classMap.get(sp));
  }
  return { X: tensor(rows), y: tensor(labels) };
}

describe('iris end-to-end', () => {
  it('scaler + classifiers reach high held-out accuracy', () => {
    const { X, y } = loadIris();
    const scaler = new ml.StandardScaler().fit(X);
    const Xs = scaler.transform(X);
    const [Xtr, Xte, ytr, yte] = ml.train_test_split(Xs, y, { testSize: 0.3, randomState: 42 });

    const logistic = new ml.LogisticRegression({ maxIter: 500 }).fit(Xtr, ytr);
    expect(logistic.score(Xte, yte)).toBeGreaterThan(0.9);

    const tree = new ml.DecisionTreeClassifier({ maxDepth: 4 }).fit(Xtr, ytr);
    expect(tree.score(Xte, yte)).toBeGreaterThan(0.9);
  });

  it('PCA(2) reduces dimensionality', () => {
    const { X } = loadIris();
    const Z = new ml.PCA({ nComponents: 2 }).fit_transform(X);
    expect(Z.shape[0]).toBe(X.shape[0]);
    expect(Z.shape[1]).toBe(2);
  });
});
