import { describe, it, expect } from 'vitest';
import { EarlyStopping, Trainer } from '../../../src/lightning/index.js';
import { SimpleModel, makeData, quietTrainerOpts, fakeTrainerWithMetric } from '../_fixtures.js';

describe('EarlyStopping', () => {
  it('records the first score as the best without waiting', () => {
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 2, mode: 'min' });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    expect(es.bestScore).toBe(1.0);
    expect(es.waitCount).toBe(0);
  });

  it('resets wait count when the score improves (min mode)', () => {
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 3, mode: 'min' });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    expect(es.waitCount).toBe(1);
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 0.5));
    expect(es.waitCount).toBe(0);
    expect(es.bestScore).toBe(0.5);
  });

  it('stops training once patience is exceeded', () => {
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 2, mode: 'min' });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    const t2 = fakeTrainerWithMetric('val_loss', 1.0);
    es.onValidationEnd(t2);
    expect(t2.shouldStop).toBe(false);
    const t3 = fakeTrainerWithMetric('val_loss', 1.0);
    es.onValidationEnd(t3);
    expect(t3.shouldStop).toBe(true);
  });

  it('max mode treats larger values as improvements', () => {
    const es = new EarlyStopping({ monitor: 'acc', patience: 5, mode: 'max' });
    es.onValidationEnd(fakeTrainerWithMetric('acc', 0.5));
    es.onValidationEnd(fakeTrainerWithMetric('acc', 0.7));
    expect(es.bestScore).toBe(0.7);
    expect(es.waitCount).toBe(0);
  });

  it('minDelta requires improvement to exceed the margin', () => {
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 5, mode: 'min', minDelta: 0.1 });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 0.95));
    expect(es.waitCount).toBe(1);
    expect(es.bestScore).toBe(1.0);
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 0.85));
    expect(es.bestScore).toBe(0.85);
  });

  it('ignores validation end when checkOnTrainEpochEnd is set', () => {
    const es = new EarlyStopping({ monitor: 'val_loss', checkOnTrainEpochEnd: true });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    expect(es.bestScore).toBeNull();
    es.onTrainEpochEnd(fakeTrainerWithMetric('val_loss', 1.0));
    expect(es.bestScore).toBe(1.0);
  });

  it('does nothing when the monitored metric is absent', () => {
    const es = new EarlyStopping({ monitor: 'missing', patience: 1 });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    expect(es.bestScore).toBeNull();
  });

  it('reset clears wait count and best score', () => {
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 2 });
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    es.onValidationEnd(fakeTrainerWithMetric('val_loss', 1.0));
    es.reset();
    expect(es.waitCount).toBe(0);
    expect(es.bestScore).toBeNull();
  });

  it('stops a real training run early on a non-improving metric', async () => {
    class ESModel extends SimpleModel {
      validationStep() { this.log('val_loss', 1.0); }
    }
    const es = new EarlyStopping({ monitor: 'val_loss', patience: 2, mode: 'min' });
    const trainer = new Trainer({ maxEpochs: 100, callbacks: [es], ...quietTrainerOpts });
    await trainer.fit(new ESModel(), makeData(20, 10), makeData(10, 10));
    expect(trainer.state.epoch).toBeLessThan(100);
  });
});
