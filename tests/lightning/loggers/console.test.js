import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ConsoleLogger } from '../../../src/lightning/index.js';

describe('ConsoleLogger', () => {
  let spy;
  beforeEach(() => { spy = vi.spyOn(console, 'log').mockImplementation(() => {}); });
  afterEach(() => { spy.mockRestore(); });

  it('emits metrics sorted by key with the step prefix', () => {
    new ConsoleLogger().logMetrics({ zebra: 1, alpha: 2 }, 5);
    const line = spy.mock.calls[0][0];
    expect(line.indexOf('alpha')).toBeLessThan(line.indexOf('zebra'));
    expect(line).toContain('[step 5]');
  });

  it('formats non-integer values to 4 decimals and keeps integers bare', () => {
    new ConsoleLogger().logMetrics({ loss: 0.123456, n: 3 }, 0);
    const line = spy.mock.calls[0][0];
    expect(line).toContain('loss: 0.1235');
    expect(line).toContain('n: 3');
  });

  it('uses exponential notation for very small magnitudes', () => {
    new ConsoleLogger().logMetrics({ lr: 0.0000123 }, 0);
    expect(spy.mock.calls[0][0]).toMatch(/lr: 1\.230e-5/);
  });

  it('logFrequency gates how often metrics are printed', () => {
    const logger = new ConsoleLogger({ logFrequency: 2 });
    logger.logMetrics({ a: 1 }, 0);
    logger.logMetrics({ a: 1 }, 1);
    logger.logMetrics({ a: 1 }, 2);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('logHyperparams prints params sorted by key', () => {
    new ConsoleLogger().logHyperparams({ lr: 0.01, batch: 32 });
    const line = spy.mock.calls[0][0];
    expect(line).toContain('[hyperparams]');
    expect(line.indexOf('batch')).toBeLessThan(line.indexOf('lr'));
  });
});
