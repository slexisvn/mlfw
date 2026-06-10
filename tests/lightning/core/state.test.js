import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import { MetricAccumulator, TrainerState } from '../../../src/lightning/index.js';

describe('MetricAccumulator', () => {
  it('mean reduction divides sum by count', () => {
    const acc = new MetricAccumulator();
    acc.update('loss', 1.0);
    acc.update('loss', 2.0);
    acc.update('loss', 6.0);
    expect(acc.compute('loss')).toBeCloseTo(3.0);
  });

  it('sum reduction accumulates total', () => {
    const acc = new MetricAccumulator();
    acc.update('n', 5, 'sum');
    acc.update('n', 3, 'sum');
    acc.update('n', 2, 'sum');
    expect(acc.compute('n')).toBe(10);
  });

  it('min and max track extremes regardless of update order', () => {
    const lo = new MetricAccumulator();
    [5, 2, 8, 1, 9].forEach(v => lo.update('v', v, 'min'));
    expect(lo.compute('v')).toBe(1);

    const hi = new MetricAccumulator();
    [5, 2, 8, 1, 9].forEach(v => hi.update('v', v, 'max'));
    expect(hi.compute('v')).toBe(9);
  });

  it('last reduction returns most recent value', () => {
    const acc = new MetricAccumulator();
    acc.update('v', 1, 'last');
    acc.update('v', 7, 'last');
    acc.update('v', 4, 'last');
    expect(acc.compute('v')).toBe(4);
  });

  it('reduction fn is locked in on first update for a name', () => {
    const acc = new MetricAccumulator();
    acc.update('v', 10, 'sum');
    acc.update('v', 5, 'mean');
    expect(acc.compute('v')).toBe(15);
  });

  it('unwraps tensor values via .item()', () => {
    const acc = new MetricAccumulator();
    acc.update('v', tensor([4]));
    acc.update('v', tensor([8]));
    expect(acc.compute('v')).toBeCloseTo(6);
  });

  it('compute returns undefined for unknown metric', () => {
    expect(new MetricAccumulator().compute('missing')).toBeUndefined();
  });

  it('computeAll reduces every tracked metric independently', () => {
    const acc = new MetricAccumulator();
    acc.update('mean_m', 2);
    acc.update('mean_m', 4);
    acc.update('sum_m', 3, 'sum');
    acc.update('sum_m', 4, 'sum');
    expect(acc.computeAll()).toEqual({ mean_m: 3, sum_m: 7 });
  });

  it('reset clears all state and size', () => {
    const acc = new MetricAccumulator();
    acc.update('a', 1);
    acc.update('b', 2);
    expect(acc.size).toBe(2);
    expect(acc.has('a')).toBe(true);
    acc.reset();
    expect(acc.size).toBe(0);
    expect(acc.has('a')).toBe(false);
    expect(acc.compute('a')).toBeUndefined();
  });
});

describe('TrainerState', () => {
  it('resetEpochMetrics clears only the epoch accumulator', () => {
    const state = new TrainerState();
    state.epochMetrics.update('a', 1);
    state.stepMetrics.update('b', 2);
    state.resetEpochMetrics();
    expect(state.epochMetrics.size).toBe(0);
    expect(state.stepMetrics.size).toBe(1);
  });

  it('resetStepMetrics clears only the step accumulator', () => {
    const state = new TrainerState();
    state.epochMetrics.update('a', 1);
    state.stepMetrics.update('b', 2);
    state.resetStepMetrics();
    expect(state.stepMetrics.size).toBe(0);
    expect(state.epochMetrics.size).toBe(1);
  });
});
