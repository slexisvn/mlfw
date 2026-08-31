import type { ScheduleArgs, SerializedStep } from '../support/trace.js';

export type { ScheduleArgs, SerializedStep };
export type ReplayTarget = Record<string, unknown> & { _replaying?: boolean; _minted?: string[] };
export type ReplayObserver = (step: ScheduleStep, index: number) => void;

const NO_OBSERVER: ReplayObserver = () => {};

function rename(arg: unknown, names: ReadonlyMap<string, string>): unknown {
  if (typeof arg === 'string') return names.get(arg) ?? arg;
  if (Array.isArray(arg)) return arg.map(item => rename(item, names));
  return arg;
}

export class ScheduleStep {
  primitive: string;
  args: ScheduleArgs;
  produced: readonly string[];

  constructor(primitive: string, args: ScheduleArgs, produced: readonly string[] = []) {
    this.primitive = primitive;
    this.args = args;
    this.produced = produced;
  }

  serialize(): SerializedStep {
    return { primitive: this.primitive, args: this.args, produced: this.produced };
  }

  static deserialize(obj: SerializedStep): ScheduleStep {
    return new ScheduleStep(obj.primitive, obj.args, obj.produced ?? []);
  }
}

export class ScheduleTrace {
  steps: ScheduleStep[];

  constructor() {
    this.steps = [];
  }

  record(primitive: string, args: ScheduleArgs, produced: readonly string[] = []): ScheduleStep {
    const step = new ScheduleStep(primitive, args, produced);
    this.steps.push(step);
    return step;
  }

  serialize(): SerializedStep[] {
    return this.steps.map(s => s.serialize());
  }

  static deserialize(arr: readonly SerializedStep[]): ScheduleTrace {
    const trace = new ScheduleTrace();
    for (const obj of arr) {
      trace.steps.push(ScheduleStep.deserialize(obj));
    }
    return trace;
  }

  replay(schedule: ReplayTarget): void {
    this.replayEach(schedule, NO_OBSERVER);
  }

  replayEach(schedule: ReplayTarget, after: ReplayObserver): void {
    const names = new Map<string, string>();

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      const method = schedule[step.primitive] as ((...args: unknown[]) => unknown) | undefined;
      if (typeof method !== 'function') {
        throw new Error(`Unknown schedule primitive: ${step.primitive}`);
      }

      const minted = schedule._minted;
      const mark = minted ? minted.length : 0;
      schedule._replaying = true;
      try {
        method.call(schedule, ...step.args.map(arg => rename(arg, names)));
      } finally {
        schedule._replaying = false;
      }

      if (minted) {
        const fresh = minted.slice(mark);
        for (let k = 0; k < step.produced.length && k < fresh.length; k++) {
          names.set(step.produced[k], fresh[k]);
        }
      }

      after(step, i);
    }
  }

  get length(): number {
    return this.steps.length;
  }

  clear(): void {
    this.steps.length = 0;
  }
}
