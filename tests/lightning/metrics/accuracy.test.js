import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import { Accuracy } from '../../../src/lightning/index.js';

describe('Accuracy', () => {
  it('binary thresholds at 0.5 by default', () => {
    const acc = new Accuracy({ task: 'binary' });
    acc.update(tensor([0.9, 0.1, 0.8, 0.3]), tensor([1, 0, 1, 0]));
    expect(acc.compute()).toBe(1.0);
  });

  it('binary respects a custom threshold', () => {
    const acc = new Accuracy({ task: 'binary', threshold: 0.8 });
    acc.update(tensor([0.75, 0.85]), tensor([1, 1]));
    expect(acc.compute()).toBe(0.5);
  });

  it('multiclass picks the argmax row', () => {
    const acc = new Accuracy({ task: 'multiclass', numClasses: 3 });
    const preds = tensor([
      0.9, 0.05, 0.05,
      0.1, 0.2, 0.7,
      0.3, 0.6, 0.1,
    ], { shape: [3, 3] });
    acc.update(preds, tensor([0, 2, 0]));
    expect(acc.compute()).toBeCloseTo(2 / 3);
  });

  it('multiclass topK counts a hit anywhere in the top K', () => {
    const acc = new Accuracy({ task: 'multiclass', numClasses: 3, topK: 2 });
    const preds = tensor([
      0.5, 0.4, 0.1,
      0.1, 0.2, 0.7,
    ], { shape: [2, 3] });
    acc.update(preds, tensor([1, 1]));
    expect(acc.compute()).toBe(1.0);
  });

  it('accumulates correct/total across multiple updates', () => {
    const acc = new Accuracy({ task: 'binary' });
    acc.update(tensor([0.9, 0.1]), tensor([1, 1]));
    acc.update(tensor([0.9, 0.9]), tensor([1, 1]));
    expect(acc.compute()).toBe(0.75);
  });

  it('reset clears the running counts', () => {
    const acc = new Accuracy({ task: 'binary' });
    acc.update(tensor([0.1]), tensor([1]));
    acc.reset();
    expect(acc.compute()).toBe(0);
    acc.update(tensor([0.9]), tensor([1]));
    expect(acc.compute()).toBe(1.0);
  });
});
