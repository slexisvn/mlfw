import { describe, it, expect } from 'vitest';
import { Trainer, LearningRateMonitor } from '../../../src/lightning/index.js';
import { SimpleModel, makeData, quietTrainerOpts } from '../_fixtures.js';

describe('LearningRateMonitor', () => {
  it('records an lr entry per training batch under the "lr" key', async () => {
    const lrm = new LearningRateMonitor();
    const trainer = new Trainer({ maxEpochs: 2, callbacks: [lrm], ...quietTrainerOpts });
    await trainer.fit(new SimpleModel(), makeData(20, 10));
    expect(lrm.lrHistory.lr).toHaveLength(4);
    lrm.lrHistory.lr.forEach(entry => {
      expect(entry).toHaveProperty('step');
      expect(entry.lr).toBeCloseTo(0.05);
    });
  });
});
