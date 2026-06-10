import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import { MeanMetric, SumMetric } from '../../../src/lightning/index.js';

describe('MeanMetric', () => {
  it('computes unweighted average across updates', () => {
    const m = new MeanMetric();
    [2, 4, 9].forEach(v => m.update(v));
    expect(m.compute()).toBeCloseTo(5);
  });

  it('applies per-sample weights to both sum and count', () => {
    const m = new MeanMetric();
    m.update(10, 3);
    m.update(2, 1);
    expect(m.compute()).toBeCloseTo(8);
  });

  it('returns 0 when empty rather than NaN', () => {
    expect(new MeanMetric().compute()).toBe(0);
  });

  it('reset restores the empty state', () => {
    const m = new MeanMetric();
    m.update(5);
    m.reset();
    expect(m.compute()).toBe(0);
  });

  it('forward updates then returns the running compute', () => {
    const m = new MeanMetric();
    m.update(2);
    expect(m.forward(4)).toBeCloseTo(3);
    expect(m.value).toBeCloseTo(3);
  });
});

describe('SumMetric', () => {
  it('accumulates a running total and unwraps tensors', () => {
    const m = new SumMetric();
    m.update(2);
    m.update(tensor([5]));
    expect(m.compute()).toBeCloseTo(7);
  });

  it('reset zeroes the total', () => {
    const m = new SumMetric();
    m.update(9);
    m.reset();
    expect(m.compute()).toBe(0);
  });
});
