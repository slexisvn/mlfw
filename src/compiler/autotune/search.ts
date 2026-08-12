import type { ScheduleSketch, SketchParams, Rng } from './sketch.js';
import type { Deadline } from './budget.js';

export type EvalResult = { score: number } | null | undefined;
export type Evaluator = (sketch: ScheduleSketch, params: SketchParams) => EvalResult;
export type PopulationMember = { sketch: ScheduleSketch; params: SketchParams };
export type SearchResult = { candidates: SearchCandidate[]; population: PopulationMember[] | null };
export type MutatorCtx = { rngFloat: () => number; rng: Rng; mutationRate: number };
export type Mutator = (params: SketchParams, sketch: ScheduleSketch, ctx: MutatorCtx) => SketchParams;
export type SearchConfig = Readonly<{
  strategy?: string;
  numTrials?: number;
  populationSize?: number;
  numGenerations?: number;
  mutationRate?: number;
  eliteRatio?: number;
  seed?: number;
  deadline?: Deadline | null;
}>;

function nextLcg(state: number): number {
  return (state * 1664525 + 1013904223) & 0x7fffffff;
}

export class SearchCandidate {
  sketchName: string;
  params: SketchParams;
  score: number;

  constructor(sketchName: string, params: SketchParams, score: number) {
    this.sketchName = sketchName;
    this.params = params;
    this.score = score;
  }
}

export class RandomSearch {
  numTrials: number;
  seed: number;
  deadline: Deadline | null;
  private _rngState: number;

  constructor(config: SearchConfig = {}) {
    this.numTrials = config.numTrials || 64;
    this.seed = config.seed || 42;
    this.deadline = config.deadline || null;
    this._rngState = this.seed;
  }

  _rng(max: number): number {
    this._rngState = nextLcg(this._rngState);
    return this._rngState % max;
  }

  _expired(): boolean {
    return this.deadline ? this.deadline.expired : false;
  }

  search(sketches: readonly ScheduleSketch[], evaluator: Evaluator): SearchResult {
    const candidates: SearchCandidate[] = [];

    for (const sketch of sketches) {
      if (this._expired()) break;
      for (let i = 0; i < this.numTrials; i++) {
        if (this._expired()) break;
        const params = sketch.sampleParams((max: number) => this._rng(max));
        const result = evaluator(sketch, params);
        if (result) {
          candidates.push(new SearchCandidate(sketch.name, params, result.score));
        }
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    return { candidates, population: null };
  }
}

export class EvolutionarySearch {
  populationSize: number;
  numGenerations: number;
  mutationRate: number;
  eliteRatio: number;
  seed: number;
  deadline: Deadline | null;
  private _rngState: number;

  constructor(config: SearchConfig = {}) {
    this.populationSize = config.populationSize || 32;
    this.numGenerations = config.numGenerations || 10;
    this.mutationRate = config.mutationRate || 0.3;
    this.eliteRatio = config.eliteRatio || 0.2;
    this.seed = config.seed || 42;
    this.deadline = config.deadline || null;
    this._rngState = this.seed;
  }

  _rng(max: number): number {
    this._rngState = nextLcg(this._rngState);
    return this._rngState % max;
  }

  _rngFloat(): number {
    this._rngState = nextLcg(this._rngState);
    return this._rngState / 0x7fffffff;
  }

  _expired(): boolean {
    return this.deadline ? this.deadline.expired : false;
  }

  search(sketches: readonly ScheduleSketch[], evaluator: Evaluator, seedPopulation: PopulationMember[] | null = null): SearchResult {
    let population = seedPopulation && seedPopulation.length ? seedPopulation : this._initPopulation(sketches);

    const evalCache = new Map<string, EvalResult>();
    const evalMemo = (sketch: ScheduleSketch, params: SketchParams): EvalResult => {
      const key = sketch.name + '|' + JSON.stringify(params);
      if (evalCache.has(key)) return evalCache.get(key);
      const r = evaluator(sketch, params);
      evalCache.set(key, r);
      return r;
    };

    for (let gen = 0; gen < this.numGenerations; gen++) {
      if (this._expired()) break;
      const scored = [];
      for (const individual of population) {
        const result = evalMemo(individual.sketch, individual.params);
        if (result) {
          scored.push({ ...individual, score: result.score });
        }
      }

      scored.sort((a, b) => b.score - a.score);
      if (scored.length === 0) break;

      const eliteCount = Math.max(1, Math.floor(scored.length * this.eliteRatio));
      const elites = scored.slice(0, eliteCount);

      const nextGen = elites.map(e => ({ sketch: e.sketch, params: { ...e.params } }));

      while (nextGen.length < this.populationSize) {
        const parentA = elites[this._rng(elites.length)];
        const parentB = elites[this._rng(elites.length)];
        if (parentA.sketch !== parentB.sketch) {
          const pick = this._rngFloat() < 0.5 ? parentA : parentB;
          nextGen.push(this._mutate(pick.sketch, { ...pick.params }));
        } else {
          nextGen.push(this._mutate(parentA.sketch, this._crossover(parentA, parentB)));
        }
      }

      population = nextGen;
    }

    const finalScored = [];
    for (const individual of population) {
      const result = evalMemo(individual.sketch, individual.params);
      if (result) {
        finalScored.push(new SearchCandidate(individual.sketch.name, individual.params, result.score));
      }
    }

    finalScored.sort((a, b) => b.score - a.score);
    return { candidates: finalScored, population };
  }

  _initPopulation(sketches: readonly ScheduleSketch[]): PopulationMember[] {
    const pop: PopulationMember[] = [];
    for (let i = 0; i < this.populationSize; i++) {
      const sketch = sketches[this._rng(sketches.length)];
      const params = sketch.sampleParams((max: number) => this._rng(max));
      pop.push({ sketch, params });
    }
    return pop;
  }

  _crossover(a: PopulationMember, b: PopulationMember): SketchParams {
    if (a.sketch !== b.sketch) return { ...a.params };
    const params: SketchParams = {};
    for (const v of a.sketch.variables) {
      params[v.name] = this._rngFloat() < 0.5 ? a.params[v.name] : b.params[v.name];
    }
    return params;
  }

  _mutate(sketch: ScheduleSketch, params: SketchParams): PopulationMember {
    const ctx: MutatorCtx = { rngFloat: () => this._rngFloat(), rng: (m: number) => this._rng(m), mutationRate: this.mutationRate };
    const muts = [defaultResampleMutator, ..._mutators];
    let cur = params;
    for (const m of muts) cur = m(cur, sketch, ctx);
    return { sketch, params: cur };
  }
}

const _mutators: Mutator[] = [];

export function registerMutator(fn: Mutator): void {
  _mutators.push(fn);
}

function defaultResampleMutator(params: SketchParams, sketch: ScheduleSketch, ctx: MutatorCtx): SketchParams {
  const newParams = { ...params };
  for (const v of sketch.variables) {
    if (ctx.rngFloat() < ctx.mutationRate) {
      newParams[v.name] = v.sample((max: number) => ctx.rng(max));
    }
  }
  return newParams;
}

export function createSearchStrategy(config: SearchConfig = {}): RandomSearch | EvolutionarySearch {
  if (config.strategy === 'random') {
    return new RandomSearch({ numTrials: config.numTrials, seed: config.seed, deadline: config.deadline });
  }
  return new EvolutionarySearch({
    populationSize: config.populationSize,
    numGenerations: config.numGenerations,
    mutationRate: config.mutationRate,
    eliteRatio: config.eliteRatio,
    seed: config.seed,
    deadline: config.deadline
  });
}
