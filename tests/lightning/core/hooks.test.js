import { describe, it, expect, vi } from 'vitest';
import { tensor } from '../../../src/index.js';
import {
  Callback, CallbackConnector, LoggerConnector, TrainerState,
} from '../../../src/lightning/index.js';

describe('CallbackConnector', () => {
  it('invokes the named hook on every callback with forwarded args', () => {
    const seen = [];
    const cb = new (class extends Callback {
      onTrainBatchEnd(trainer, model, outputs) { seen.push(outputs); }
    })();
    const conn = new CallbackConnector([cb]);
    conn.dispatch('onTrainBatchEnd', {}, {}, 42);
    expect(seen).toEqual([42]);
  });

  it('binds `this` to the callback instance', () => {
    let captured = null;
    const cb = new (class extends Callback {
      constructor() { super(); this.tag = 'me'; }
      onFitStart() { captured = this.tag; }
    })();
    new CallbackConnector([cb]).dispatch('onFitStart');
    expect(captured).toBe('me');
  });

  it('skips callbacks that do not define the hook', () => {
    const plain = { onFitStart() { throw new Error('should be reachable'); } };
    const noHook = {};
    const conn = new CallbackConnector([noHook, plain]);
    expect(() => conn.dispatch('onFitEnd')).not.toThrow();
  });

  it('add and remove mutate the callback list', () => {
    const conn = new CallbackConnector([]);
    const cb = new Callback();
    conn.add(cb);
    expect(conn.callbacks).toContain(cb);
    conn.remove(cb);
    expect(conn.callbacks).not.toContain(cb);
  });

  it('dispatches to callbacks in registration order', () => {
    const order = [];
    const mk = tag => ({ onFitStart() { order.push(tag); } });
    new CallbackConnector([mk('a'), mk('b'), mk('c')]).dispatch('onFitStart');
    expect(order).toEqual(['a', 'b', 'c']);
  });
});

describe('LoggerConnector', () => {
  it('routes onEpoch entries to epoch metrics and onStep entries to step metrics', () => {
    const state = new TrainerState();
    const conn = new LoggerConnector([], state);
    const model = { _logBuffer: new Map() };
    model._logBuffer.set('e', { value: 5, onStep: false, onEpoch: true, reduceFx: 'mean', progBar: false });
    model._logBuffer.set('s', { value: 7, onStep: true, onEpoch: false, reduceFx: 'mean', progBar: false });

    conn.drain(model);

    expect(state.epochMetrics.compute('e')).toBe(5);
    expect(state.epochMetrics.has('s')).toBe(false);
    expect(state.stepMetrics.compute('s')).toBe(7);
    expect(model._logBuffer.size).toBe(0);
  });

  it('records progBar metrics into a side map, unwrapping tensors', () => {
    const state = new TrainerState();
    const conn = new LoggerConnector([], state);
    const model = { _logBuffer: new Map() };
    model._logBuffer.set('p', { value: tensor([3]), onStep: false, onEpoch: false, reduceFx: 'mean', progBar: true });
    conn.drain(model);
    expect(state._progBarMetrics.get('p')).toBeCloseTo(3);
  });

  it('flushStepMetrics computes, forwards to loggers, and resets', () => {
    const state = new TrainerState();
    const logger = { logMetrics: vi.fn() };
    const conn = new LoggerConnector([logger], state);
    state.stepMetrics.update('loss', 2);
    state.stepMetrics.update('loss', 4);

    const out = conn.flushStepMetrics(10);

    expect(out.loss).toBe(3);
    expect(logger.logMetrics).toHaveBeenCalledWith({ loss: 3 }, 10);
    expect(state.stepMetrics.size).toBe(0);
  });

  it('flush is a no-op when there are no metrics', () => {
    const logger = { logMetrics: vi.fn() };
    const conn = new LoggerConnector([logger], new TrainerState());
    const out = conn.flushEpochMetrics(0);
    expect(out).toEqual({});
    expect(logger.logMetrics).not.toHaveBeenCalled();
  });

  it('drain on an empty buffer does nothing', () => {
    const state = new TrainerState();
    const conn = new LoggerConnector([], state);
    conn.drain({ _logBuffer: new Map() });
    expect(state.epochMetrics.size).toBe(0);
  });
});
