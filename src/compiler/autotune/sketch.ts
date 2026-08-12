import type { Schedule } from '../schedule/schedule.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

export type Rng = (n: number) => number;
export type SketchParamValue = number | readonly number[];
export type SketchParams = Record<string, SketchParamValue>;
export type SketchApplyFn = (schedule: Schedule, blockName: string, target: ScheduleTarget, params: SketchParams) => void;
export type SketchInstance = (schedule: Schedule, blockName: string, target: ScheduleTarget) => void;

export class SearchVariable {
  name: string;
  candidates: readonly SketchParamValue[];

  constructor(name: string, candidates: readonly SketchParamValue[]) {
    this.name = name;
    this.candidates = candidates;
  }

  sample(rng: Rng): SketchParamValue {
    return this.candidates[rng(this.candidates.length)];
  }
}

export class ScheduleSketch {
  name: string;
  variables: readonly SearchVariable[];
  private _apply: SketchApplyFn;

  constructor(name: string, variables: readonly SearchVariable[], apply: SketchApplyFn) {
    this.name = name;
    this.variables = variables;
    this._apply = apply;
  }

  instantiate(params: SketchParams): SketchInstance {
    return (schedule: Schedule, blockName: string, target: ScheduleTarget) => {
      this._apply(schedule, blockName, target, params);
    };
  }

  sampleParams(rng: Rng): SketchParams {
    const params: SketchParams = {};
    for (const v of this.variables) {
      params[v.name] = v.sample(rng);
    }
    return params;
  }
}
