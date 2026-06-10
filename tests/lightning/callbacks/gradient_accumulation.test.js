import { describe, it, expect } from 'vitest';
import { GradientAccumulationScheduler } from '../../../src/lightning/index.js';

describe('GradientAccumulationScheduler', () => {
  it('returns the accumulation in effect for a given epoch (step function)', () => {
    const s = new GradientAccumulationScheduler({ scheduling: { 0: 4, 2: 2, 4: 1 } });
    expect(s.getCurrentAccumulation(0)).toBe(4);
    expect(s.getCurrentAccumulation(1)).toBe(4);
    expect(s.getCurrentAccumulation(2)).toBe(2);
    expect(s.getCurrentAccumulation(3)).toBe(2);
    expect(s.getCurrentAccumulation(4)).toBe(1);
    expect(s.getCurrentAccumulation(99)).toBe(1);
  });

  it('defaults to 1 before the first scheduled epoch', () => {
    const s = new GradientAccumulationScheduler({ scheduling: { 5: 8 } });
    expect(s.getCurrentAccumulation(0)).toBe(1);
    expect(s.getCurrentAccumulation(4)).toBe(1);
    expect(s.getCurrentAccumulation(5)).toBe(8);
  });

  it('applies the scheduled value to the trainer at epoch start', () => {
    const s = new GradientAccumulationScheduler({ scheduling: { 0: 4, 2: 2 } });
    const trainer = { accumulateGradBatches: 1, state: { epoch: 2 } };
    s.onTrainEpochStart(trainer);
    expect(trainer.accumulateGradBatches).toBe(2);
  });

  it('leaves the trainer untouched on an unscheduled epoch', () => {
    const s = new GradientAccumulationScheduler({ scheduling: { 0: 4 } });
    const trainer = { accumulateGradBatches: 4, state: { epoch: 1 } };
    s.onTrainEpochStart(trainer);
    expect(trainer.accumulateGradBatches).toBe(4);
  });
});
