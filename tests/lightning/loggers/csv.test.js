import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CSVLogger } from '../../../src/lightning/index.js';

describe('CSVLogger', () => {
  let dir;
  beforeEach(() => { dir = mkdtempSync(join(tmpdir(), 'csvlog-')); });
  afterEach(() => { rmSync(dir, { recursive: true, force: true }); });

  it('writes a header plus rows on finalize', () => {
    const logger = new CSVLogger({ saveDir: dir, version: 0 });
    logger.logMetrics({ loss: 0.5, acc: 1 }, 0);
    logger.logMetrics({ loss: 0.4, acc: 1 }, 1);
    logger.finalize();

    const csv = readFileSync(join(logger.logDir, 'metrics.csv'), 'utf8').trim().split('\n');
    expect(csv[0]).toBe('step,loss,acc');
    expect(csv[1]).toBe('0,0.5,1');
    expect(csv[2]).toBe('1,0.4,1');
  });

  it('extends the schema when a new column appears later', () => {
    const logger = new CSVLogger({ saveDir: dir, version: 0 });
    logger.logMetrics({ a: 1 }, 0);
    logger.logMetrics({ a: 2, b: 9 }, 1);
    logger.finalize();

    const csv = readFileSync(join(logger.logDir, 'metrics.csv'), 'utf8').trim().split('\n');
    expect(csv[0]).toBe('step,a,b');
    expect(csv[1]).toBe('0,1,');
    expect(csv[2]).toBe('1,2,9');
  });

  it('flushes automatically once the buffer hits flushInterval', () => {
    const logger = new CSVLogger({ saveDir: dir, version: 0, flushInterval: 2 });
    logger.logMetrics({ a: 1 }, 0);
    expect(existsSync(join(logger.logDir, 'metrics.csv'))).toBe(false);
    logger.logMetrics({ a: 2 }, 1);
    expect(existsSync(join(logger.logDir, 'metrics.csv'))).toBe(true);
  });

  it('writes hyperparameters as JSON', () => {
    const logger = new CSVLogger({ saveDir: dir, version: 0 });
    logger.logHyperparams({ lr: 0.01, epochs: 3 });
    const hp = JSON.parse(readFileSync(join(logger.logDir, 'hparams.json'), 'utf8'));
    expect(hp).toEqual({ lr: 0.01, epochs: 3 });
  });
});
