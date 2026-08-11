import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import { ConfusionMatrix } from '../../../src/lightning/index.js';

describe('ConfusionMatrix', () => {
  it('places argmax predictions at [target][pred]', () => {
    const cm = new ConfusionMatrix({ numClasses: 2 });
    const preds = tensor([
      0.9, 0.1,
      0.2, 0.8,
      0.7, 0.3,
    ], { shape: [3, 2] });
    cm.update(preds, tensor([0, 1, 0]));
    const m = cm.compute();
    expect(m[0][0]).toBe(2);
    expect(m[1][1]).toBe(1);
    expect(m[0][1]).toBe(0);
    expect(m[1][0]).toBe(0);
  });

  it('counts misclassifications off the diagonal', () => {
    const cm = new ConfusionMatrix({ numClasses: 2 });
    cm.update(tensor([0.2, 0.8], { shape: [1, 2] }), tensor([0]));
    const m = cm.compute();
    expect(m[0][1]).toBe(1);
    expect(m[0][0]).toBe(0);
  });

  it('uses integer class indices when preds are 1-D', () => {
    const cm = new ConfusionMatrix({ numClasses: 3 });
    cm.update(tensor([0, 2, 1]), tensor([0, 1, 1]));
    const m = cm.compute();
    expect(m[0][0]).toBe(1);
    expect(m[1][2]).toBe(1);
    expect(m[1][1]).toBe(1);
  });

  it('accumulates across updates and resets', () => {
    const cm = new ConfusionMatrix({ numClasses: 2 });
    cm.update(tensor([0, 1]), tensor([0, 1]));
    cm.update(tensor([0]), tensor([0]));
    expect(cm.compute()[0][0]).toBe(2);
    cm.reset();
    expect(cm.compute()[0][0]).toBe(0);
  });
});
