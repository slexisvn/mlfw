import { describe, it, expect } from 'vitest';
import {
  Trainer, ProgressCallback, ModelCheckpoint, ConsoleLogger,
} from '../../../src/lightning/index.js';
import { quietTrainerOpts } from '../_fixtures.js';

describe('Trainer construction', () => {
  it('fastDevRun=true limits all splits to 1 batch and 1 epoch', () => {
    const t = new Trainer({ fastDevRun: true, ...quietTrainerOpts, maxEpochs: 50 });
    expect(t.limitTrainBatches).toBe(1);
    expect(t.limitValBatches).toBe(1);
    expect(t.limitTestBatches).toBe(1);
    expect(t.state.maxEpochs).toBe(1);
  });

  it('fastDevRun=N limits all splits to N batches', () => {
    const t = new Trainer({ fastDevRun: 3, ...quietTrainerOpts });
    expect(t.limitTrainBatches).toBe(3);
    expect(t.limitValBatches).toBe(3);
  });

  it('logger:false resolves to no loggers', () => {
    expect(new Trainer({ logger: false }).loggers).toHaveLength(0);
  });

  it('logger:true resolves to a single ConsoleLogger', () => {
    const loggers = new Trainer({ logger: true, enableProgress: false, enableCheckpointing: false }).loggers;
    expect(loggers).toHaveLength(1);
    expect(loggers[0]).toBeInstanceOf(ConsoleLogger);
  });

  it('a custom logger instance is wrapped in an array', () => {
    const custom = new ConsoleLogger();
    expect(new Trainer({ logger: custom, enableProgress: false, enableCheckpointing: false }).loggers[0]).toBe(custom);
  });

  it('auto-adds ProgressCallback by default but not ModelCheckpoint', () => {
    const t = new Trainer({ logger: false });
    expect(t.callbacks.some(c => c instanceof ProgressCallback)).toBe(true);
    expect(t.callbacks.some(c => c instanceof ModelCheckpoint)).toBe(false);
  });

  it('auto-adds ModelCheckpoint when enableCheckpointing is true', () => {
    const t = new Trainer({ logger: false, enableCheckpointing: true });
    expect(t.callbacks.some(c => c instanceof ModelCheckpoint)).toBe(true);
  });

  it('does not duplicate a user-supplied ProgressCallback', () => {
    const t = new Trainer({ callbacks: [new ProgressCallback()], enableCheckpointing: false, logger: false });
    expect(t.callbacks.filter(c => c instanceof ProgressCallback)).toHaveLength(1);
  });

  it('omits auto callbacks when disabled', () => {
    const t = new Trainer(quietTrainerOpts);
    expect(t.callbacks.some(c => c instanceof ProgressCallback)).toBe(false);
    expect(t.callbacks.some(c => c instanceof ModelCheckpoint)).toBe(false);
  });
});
