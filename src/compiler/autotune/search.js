class SearchCandidate {
  constructor(sketchName, params, score, features) {
    this.sketchName = sketchName;
    this.params = params;
    this.score = score;
    this.features = features;
  }
}

export class RandomSearch {
  constructor(config = {}) {
    this.numTrials = config.numTrials || 64;
    this.seed = config.seed || 42;
    this.deadline = config.deadline || null;
    this._rngState = this.seed;
  }

  _rng(max) {
    this._rngState = (this._rngState * 1664525 + 1013904223) & 0x7fffffff;
    return this._rngState % max;
  }

  _expired() {
    return this.deadline ? this.deadline.expired : false;
  }

  search(sketches, evaluator) {
    const candidates = [];

    for (const sketch of sketches) {
      if (this._expired()) break;
      for (let i = 0; i < this.numTrials; i++) {
        if (this._expired()) break;
        const params = sketch.sampleParams((max) => this._rng(max));
        const result = evaluator(sketch, params);
        if (result) {
          candidates.push(new SearchCandidate(sketch.name, params, result.score, result.features));
        }
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates;
  }
}

export class EvolutionarySearch {
  constructor(config = {}) {
    this.populationSize = config.populationSize || 32;
    this.numGenerations = config.numGenerations || 10;
    this.mutationRate = config.mutationRate || 0.3;
    this.eliteRatio = config.eliteRatio || 0.2;
    this.seed = config.seed || 42;
    this.deadline = config.deadline || null;
    this._rngState = this.seed;
  }

  _rng(max) {
    this._rngState = (this._rngState * 1664525 + 1013904223) & 0x7fffffff;
    return this._rngState % max;
  }

  _rngFloat() {
    this._rngState = (this._rngState * 1664525 + 1013904223) & 0x7fffffff;
    return this._rngState / 0x7fffffff;
  }

  _expired() {
    return this.deadline ? this.deadline.expired : false;
  }

  search(sketches, evaluator) {
    let population = this._initPopulation(sketches);

    for (let gen = 0; gen < this.numGenerations; gen++) {
      if (this._expired()) break;
      const scored = [];
      for (const individual of population) {
        const result = evaluator(individual.sketch, individual.params);
        if (result) {
          scored.push({ ...individual, score: result.score, features: result.features });
        }
      }

      scored.sort((a, b) => b.score - a.score);
      if (scored.length === 0) break;

      const eliteCount = Math.max(1, Math.floor(scored.length * this.eliteRatio));
      const elites = scored.slice(0, eliteCount);

      const nextGen = elites.map(e => ({ sketch: e.sketch, params: { ...e.params } }));

      while (nextGen.length < this.populationSize) {
        const parentIdx = this._rng(elites.length);
        const parent = elites[parentIdx];
        const child = this._mutate(parent.sketch, parent.params);
        nextGen.push(child);
      }

      population = nextGen;
    }

    const finalScored = [];
    for (const individual of population) {
      const result = evaluator(individual.sketch, individual.params);
      if (result) {
        finalScored.push(new SearchCandidate(individual.sketch.name, individual.params, result.score, result.features));
      }
    }

    finalScored.sort((a, b) => b.score - a.score);
    return finalScored;
  }

  _initPopulation(sketches) {
    const pop = [];
    for (let i = 0; i < this.populationSize; i++) {
      const sketch = sketches[this._rng(sketches.length)];
      const params = sketch.sampleParams((max) => this._rng(max));
      pop.push({ sketch, params });
    }
    return pop;
  }

  _mutate(sketch, params) {
    const newParams = { ...params };
    for (const v of sketch.variables) {
      if (this._rngFloat() < this.mutationRate) {
        newParams[v.name] = v.sample((max) => this._rng(max));
      }
    }
    return { sketch, params: newParams };
  }
}
