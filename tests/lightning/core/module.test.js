import { describe, it, expect, vi } from 'vitest';
import { LightningModule } from '../../../src/lightning/index.js';
import { parseOptimizersConfig } from '../../../src/lightning/core/module.js';
import { SimpleModel } from '../_fixtures.js';

describe('LightningModule.log', () => {
  it('defaults to onStep during training stage', () => {
    const m = new SimpleModel();
    m._trainer = { state: { stage: 'training' } };
    m.log('x', 1);
    const entry = m._logBuffer.get('x');
    expect(entry.onStep).toBe(true);
    expect(entry.onEpoch).toBe(false);
  });

  it('defaults to onEpoch during non-training stage', () => {
    const m = new SimpleModel();
    m._trainer = { state: { stage: 'validating' } };
    m.log('x', 1);
    const entry = m._logBuffer.get('x');
    expect(entry.onStep).toBe(false);
    expect(entry.onEpoch).toBe(true);
  });

  it('explicit onStep/onEpoch override the stage defaults', () => {
    const m = new SimpleModel();
    m._trainer = { state: { stage: 'training' } };
    m.log('x', 1, { onStep: false, onEpoch: true });
    const entry = m._logBuffer.get('x');
    expect(entry.onStep).toBe(false);
    expect(entry.onEpoch).toBe(true);
  });

  it('logDict logs each key with shared options', () => {
    const m = new SimpleModel();
    m._trainer = { state: { stage: 'training' } };
    m.logDict({ a: 1, b: 2 }, { progBar: true });
    expect(m._logBuffer.get('a').progBar).toBe(true);
    expect(m._logBuffer.get('b').value).toBe(2);
  });
});

describe('LightningModule.manualBackward', () => {
  it('falls back to loss.backward when no strategy', () => {
    const m = new SimpleModel();
    const loss = { backward: vi.fn() };
    m.manualBackward(loss);
    expect(loss.backward).toHaveBeenCalledOnce();
  });

  it('routes through the trainer strategy when present', () => {
    const m = new SimpleModel();
    const backward = vi.fn();
    m._trainer = { strategy: { backward } };
    const loss = { backward: vi.fn() };
    m.manualBackward(loss);
    expect(backward).toHaveBeenCalledWith(loss);
    expect(loss.backward).not.toHaveBeenCalled();
  });
});

describe('LightningModule unimplemented hooks', () => {
  it('throws a descriptive error for unimplemented trainingStep', () => {
    class Bare extends LightningModule {}
    expect(() => new Bare().trainingStep()).toThrow(/trainingStep\(\) not implemented/);
  });

  it('throws for unimplemented configureOptimizers', () => {
    class Bare extends LightningModule {}
    expect(() => new Bare().configureOptimizers()).toThrow(/configureOptimizers\(\) not implemented/);
  });
});

describe('parseOptimizersConfig', () => {
  const fakeOpt = () => ({ step() {}, defaults: {} });

  it('wraps a bare optimizer (has step()) with a null scheduler', () => {
    const opt = fakeOpt();
    const { optimizers, schedulerConfigs } = parseOptimizersConfig(opt);
    expect(optimizers).toEqual([opt]);
    expect(schedulerConfigs).toEqual([null]);
  });

  it('normalizes {optimizer, lrScheduler:{scheduler}} with default interval/frequency', () => {
    const opt = fakeOpt();
    const scheduler = { step() {} };
    const { optimizers, schedulerConfigs } = parseOptimizersConfig({
      optimizer: opt,
      lrScheduler: { scheduler },
    });
    expect(optimizers).toEqual([opt]);
    expect(schedulerConfigs[0]).toMatchObject({
      scheduler,
      interval: 'epoch',
      frequency: 1,
      monitor: null,
    });
  });

  it('honors an explicit interval and monitor in the scheduler config', () => {
    const { schedulerConfigs } = parseOptimizersConfig({
      optimizer: fakeOpt(),
      lrScheduler: { scheduler: { step() {} }, interval: 'step', monitor: 'val_loss' },
    });
    expect(schedulerConfigs[0].interval).toBe('step');
    expect(schedulerConfigs[0].monitor).toBe('val_loss');
  });

  it('treats a bare scheduler (has step) as an epoch scheduler', () => {
    const scheduler = { step() {} };
    const { schedulerConfigs } = parseOptimizersConfig({
      optimizer: fakeOpt(),
      lrScheduler: scheduler,
    });
    expect(schedulerConfigs[0]).toMatchObject({ scheduler, interval: 'epoch' });
  });

  it('handles an array of optimizers with mixed scheduler presence', () => {
    const a = fakeOpt();
    const b = fakeOpt();
    const sched = { step() {} };
    const { optimizers, schedulerConfigs } = parseOptimizersConfig([
      { optimizer: a, lrScheduler: { scheduler: sched } },
      b,
    ]);
    expect(optimizers).toEqual([a, b]);
    expect(schedulerConfigs[0].scheduler).toBe(sched);
    expect(schedulerConfigs[1]).toBeNull();
  });

  it('throws on null/undefined', () => {
    expect(() => parseOptimizersConfig(null)).toThrow(/null\/undefined/);
  });

  it('throws on an unrecognized object', () => {
    expect(() => parseOptimizersConfig({ foo: 1 })).toThrow(/unrecognized format/);
  });
});
