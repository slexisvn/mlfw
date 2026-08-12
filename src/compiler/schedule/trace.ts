export type ScheduleArgs = readonly unknown[];
export type SerializedStep = { primitive: string; args: ScheduleArgs };
export type ReplayTarget = Record<string, unknown> & { _replaying?: boolean };

export class ScheduleStep {
  primitive: string;
  args: ScheduleArgs;

  constructor(primitive: string, args: ScheduleArgs) {
    this.primitive = primitive;
    this.args = args;
  }

  serialize(): SerializedStep {
    return { primitive: this.primitive, args: this.args };
  }

  static deserialize(obj: SerializedStep): ScheduleStep {
    return new ScheduleStep(obj.primitive, obj.args);
  }
}

export class ScheduleTrace {
  steps: ScheduleStep[];

  constructor() {
    this.steps = [];
  }

  record(primitive: string, args: ScheduleArgs): ScheduleStep {
    const step = new ScheduleStep(primitive, args);
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
    for (const step of this.steps) {
      const method = schedule[step.primitive] as ((...args: unknown[]) => unknown) | undefined;
      if (typeof method !== 'function') {
        throw new Error(`Unknown schedule primitive: ${step.primitive}`);
      }
      schedule._replaying = true;
      try {
        method.call(schedule, ...step.args);
      } finally {
        schedule._replaying = false;
      }
    }
  }

  get length(): number {
    return this.steps.length;
  }

  clear(): void {
    this.steps.length = 0;
  }
}
