import { describe, it, expect } from 'vitest';
import { Trainer, Timer } from '../../../src/lightning/index.js';
import { SimpleModel, makeData, quietTrainerOpts } from '../_fixtures.js';

describe('Timer callback', () => {
  it('records one epoch duration per epoch and a total', async () => {
    const timer = new Timer();
    const trainer = new Trainer({ maxEpochs: 2, callbacks: [timer], ...quietTrainerOpts });
    await trainer.fit(new SimpleModel(), makeData(20, 10));
    expect(timer.epochDurations).toHaveLength(2);
    expect(timer.totalTrainingTime).toBeGreaterThanOrEqual(0);
    timer.epochDurations.forEach(d => expect(d).toBeGreaterThanOrEqual(0));
  });

  it('records a validation duration per validation pass', async () => {
    const timer = new Timer();
    const trainer = new Trainer({ maxEpochs: 2, callbacks: [timer], ...quietTrainerOpts });
    await trainer.fit(new SimpleModel(), makeData(20, 10), makeData(10, 10));
    expect(timer.validationDurations).toHaveLength(2);
  });
});
