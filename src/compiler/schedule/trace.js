export class ScheduleStep {
  constructor(primitive, args) {
    this.primitive = primitive;
    this.args = args;
  }

  serialize() {
    return { primitive: this.primitive, args: this.args };
  }

  static deserialize(obj) {
    return new ScheduleStep(obj.primitive, obj.args);
  }
}

export class ScheduleTrace {
  constructor() {
    this.steps = [];
  }

  record(primitive, args) {
    const step = new ScheduleStep(primitive, args);
    this.steps.push(step);
    return step;
  }

  serialize() {
    return this.steps.map(s => s.serialize());
  }

  static deserialize(arr) {
    const trace = new ScheduleTrace();
    for (const obj of arr) {
      trace.steps.push(ScheduleStep.deserialize(obj));
    }
    return trace;
  }

  replay(schedule) {
    for (const step of this.steps) {
      const method = schedule[step.primitive];
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

  get length() {
    return this.steps.length;
  }

  clear() {
    this.steps.length = 0;
  }
}
