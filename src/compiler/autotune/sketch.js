export class SearchVariable {
  constructor(name, candidates) {
    this.name = name;
    this.candidates = candidates;
  }

  sample(rng) {
    return this.candidates[rng(this.candidates.length)];
  }
}

export class ScheduleSketch {
  constructor(name, variables, apply) {
    this.name = name;
    this.variables = variables;
    this._apply = apply;
  }

  instantiate(params) {
    return (schedule, blockName, target) => {
      this._apply(schedule, blockName, target, params);
    };
  }

  sampleParams(rng) {
    const params = {};
    for (const v of this.variables) {
      params[v.name] = v.sample(rng);
    }
    return params;
  }
}
