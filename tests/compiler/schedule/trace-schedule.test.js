import { describe, it, expect } from 'vitest';
import { ScheduleTrace } from '../../../src/compiler/schedule/trace.js';

class FakeSchedule {
  constructor() {
    this._replaying = false;
    this.calls = [];
  }

  ok(...args) {
    this.calls.push(['ok', args]);
  }

  boom() {
    throw new Error('boom');
  }
}

describe('ScheduleTrace.replay', () => {
  it('clears _replaying after each step completes', () => {
    const trace = new ScheduleTrace();
    trace.record('ok', [1]);
    trace.record('ok', [2]);
    const schedule = new FakeSchedule();

    trace.replay(schedule);

    expect(schedule._replaying).toBe(false);
    expect(schedule.calls).toEqual([['ok', [1]], ['ok', [2]]]);
  });

  it('clears _replaying even when a primitive throws', () => {
    const trace = new ScheduleTrace();
    trace.record('boom', []);
    const schedule = new FakeSchedule();

    expect(() => trace.replay(schedule)).toThrow('boom');
    expect(schedule._replaying).toBe(false);
  });

  it('throws on unknown primitive without leaving _replaying set', () => {
    const trace = new ScheduleTrace();
    trace.record('does_not_exist', []);
    const schedule = new FakeSchedule();

    expect(() => trace.replay(schedule)).toThrow(/Unknown schedule primitive/);
    expect(schedule._replaying).toBe(false);
  });
});
