import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import { Precision, Recall, F1Score } from '../../../src/lightning/index.js';

const preds = () => tensor([0.1, 0.1, 0.9, 0.1]);
const target = () => tensor([0, 0, 0, 1]);

describe('Recall', () => {
  it('macro averages per-class equally (ignores support)', () => {
    const r = new Recall({ task: 'binary', numClasses: 2, average: 'macro' });
    r.update(preds(), target());
    expect(r.compute()).toBeCloseTo(1 / 3);
  });

  it('micro weights per-class by support', () => {
    const r = new Recall({ task: 'binary', numClasses: 2, average: 'micro' });
    r.update(preds(), target());
    expect(r.compute()).toBeCloseTo(0.5);
  });

  it('average="none" returns per-class scores', () => {
    const r = new Recall({ task: 'binary', numClasses: 2, average: 'none' });
    r.update(preds(), target());
    const perClass = r.compute();
    expect(perClass).toHaveLength(2);
    expect(perClass[0]).toBeCloseTo(2 / 3);
    expect(perClass[1]).toBeCloseTo(0);
  });

  it('reset clears the confusion counters', () => {
    const r = new Recall({ task: 'binary', numClasses: 2, average: 'micro' });
    r.update(preds(), target());
    r.reset();
    r.update(tensor([0.9, 0.9]), tensor([1, 1]));
    expect(r.compute()).toBe(1.0);
  });
});

describe('Precision', () => {
  it('macro averages per-class precision', () => {
    const p = new Precision({ task: 'binary', numClasses: 2, average: 'macro' });
    p.update(preds(), target());
    expect(p.compute()).toBeCloseTo(1 / 3);
  });
});

describe('F1Score', () => {
  it('combines precision and recall harmonically', () => {
    const f1 = new F1Score({ task: 'binary', numClasses: 2, average: 'macro' });
    f1.update(preds(), target());
    expect(f1.compute()).toBeCloseTo(1 / 3);
  });

  it('perfect predictions give F1 = 1', () => {
    const f1 = new F1Score({ task: 'multiclass', numClasses: 2, average: 'macro' });
    const p = tensor([0.9, 0.1, 0.2, 0.8, 0.8, 0.2, 0.1, 0.9], { shape: [4, 2] });
    f1.update(p, tensor([0, 1, 0, 1]));
    expect(f1.compute()).toBe(1.0);
  });
});
