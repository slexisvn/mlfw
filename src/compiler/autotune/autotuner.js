import { Schedule, resetVarCounter } from '../schedule/schedule.js';
import { ScheduleValidator } from '../schedule/validator.js';
import { FeatureExtractor } from '../cost_model/features.js';
import { AnalyticalCostModel, LearnedCostModel } from '../cost_model/cost_model.js';
import { getSketchesForBlock } from './search_space.js';
import { RandomSearch, EvolutionarySearch } from './search.js';
import { TuningRecord, TuningDatabase } from './tuning_db.js';
import { BenchmarkRunner } from './benchmark.js';
import { computeWorkloadKey } from './workload_key.js';
import { PrimFunc } from '../ir/tensor/nodes.js';

export class AutotuneConfig {
  constructor(opts = {}) {
    this.strategy = opts.strategy || 'evolutionary';
    this.numTrials = opts.numTrials || 64;
    this.populationSize = opts.populationSize || 32;
    this.numGenerations = opts.numGenerations || 10;
    this.seed = opts.seed || 42;
    this.timeBudgetMs = opts.timeBudgetMs || 30000;
    this.useTuningDB = opts.useTuningDB !== false;
    this.enableBenchmark = opts.enableBenchmark ?? false;
    this.benchmarkWarmup = opts.benchmarkWarmup ?? 3;
    this.benchmarkRepeat = opts.benchmarkRepeat ?? 10;
    this.topKForBenchmark = opts.topKForBenchmark ?? 5;
  }
}

function collectBlockNames(node, result = []) {
  if (!node) return result;
  if (node.type === 'BlockNode') result.push(node.name);
  if (node.body) collectBlockNames(node.body, result);
  if (node.stmts) for (const s of node.stmts) collectBlockNames(s, result);
  if (node.thenBody) collectBlockNames(node.thenBody, result);
  if (node.elseBody) collectBlockNames(node.elseBody, result);
  if (node.initBody) collectBlockNames(node.initBody, result);
  return result;
}

function clonePrimFunc(primFunc) {
  const cloneNode = (node) => {
    if (!node || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(cloneNode);
    const copy = Object.create(Object.getPrototypeOf(node));
    for (const key of Object.keys(node)) {
      const val = node[key];
      if (val instanceof Map) {
        copy[key] = new Map(val);
      } else if (Array.isArray(val)) {
        copy[key] = val.map(cloneNode);
      } else if (typeof val === 'object' && val !== null && val.type) {
        copy[key] = cloneNode(val);
      } else {
        copy[key] = val;
      }
    }
    return copy;
  };
  return cloneNode(primFunc);
}

export class Autotuner {
  constructor(target, config = {}) {
    this.target = target;
    this.config = config instanceof AutotuneConfig ? config : new AutotuneConfig(config);
    this.costModel = new AnalyticalCostModel(target);
    this.learnedModel = new LearnedCostModel();
    this.db = new TuningDatabase();
    this.benchmarkRunner = this.config.enableBenchmark
      ? new BenchmarkRunner(target, { warmup: this.config.benchmarkWarmup, repeat: this.config.benchmarkRepeat })
      : null;
  }

  tune(primFunc, blockName = null) {
    const blockNames = blockName ? [blockName] : collectBlockNames(primFunc.body);
    const results = new Map();

    for (const name of blockNames) {
      const result = this._tuneBlock(primFunc, name);
      if (result) results.set(name, result);
    }

    return results;
  }

  tuneAndApply(primFunc, blockName = null) {
    const tuneResults = this.tune(primFunc, blockName);

    let bestFunc = primFunc;
    let bestScore = -Infinity;

    for (const [name, result] of tuneResults) {
      if (result.score > bestScore) {
        bestScore = result.score;
      }
    }

    if (tuneResults.size > 0) {
      const best = this._applyBestSchedule(primFunc, tuneResults);
      if (best) return { func: best.func, results: tuneResults, applied: true };
    }

    return { func: primFunc, results: tuneResults, applied: false };
  }

  _tuneBlock(primFunc, blockName) {
    const workloadKey = this._computeWorkloadKey(primFunc, blockName);

    if (this.config.useTuningDB && this.db.has(workloadKey)) {
      const cached = this.db.lookup(workloadKey);
      return { sketchName: cached.sketchName, params: cached.params, score: cached.score, fromCache: true };
    }

    const sketches = getSketchesForBlock(primFunc, blockName, this.target);
    if (sketches.length === 0) return null;

    const evaluator = (sketch, params) => {
      return this._evaluate(primFunc, blockName, sketch, params);
    };

    let candidates;
    if (this.config.strategy === 'evolutionary') {
      const search = new EvolutionarySearch({
        populationSize: this.config.populationSize,
        numGenerations: this.config.numGenerations,
        seed: this.config.seed
      });
      candidates = search.search(sketches, evaluator);
    } else {
      const search = new RandomSearch({
        numTrials: this.config.numTrials,
        seed: this.config.seed
      });
      candidates = search.search(sketches, evaluator);
    }

    if (candidates.length === 0) return null;

    let best = candidates[0];

    if (this.benchmarkRunner && candidates.length > 0) {
      const measured = this._refineByCostModel(primFunc, blockName, candidates);
      if (measured.length > 0) {
        best = measured[0];
      }
    }

    if (this.config.useTuningDB) {
      const record = new TuningRecord(
        workloadKey, best.sketchName, best.params,
        best.score, null, this.db.version
      );
      record.medianMs = best.medianMs || null;
      record.minMs = best.minMs || null;
      this.db.store(workloadKey, record);
    }

    return { sketchName: best.sketchName, params: best.params, score: best.score, fromCache: false, medianMs: best.medianMs, minMs: best.minMs };
  }

  _evaluate(primFunc, blockName, sketch, params) {
    try {
      const cloned = clonePrimFunc(primFunc);
      const sch = new Schedule(cloned);
      const apply = sketch.instantiate(params);
      apply(sch, blockName, this.target);

      const errors = ScheduleValidator.validate(cloned);
      if (errors.length > 0) return null;

      const features = FeatureExtractor.extract(cloned);
      const cost = this.costModel.estimateFromFeatures(features);

      return { score: cost.score, features };
    } catch (e) {
      return null;
    }
  }

  _applyBestSchedule(primFunc, tuneResults) {
    try {
      const sch = new Schedule(primFunc);
      for (const [blockName, result] of tuneResults) {
        if (result.fromCache) continue;
        const sketches = getSketchesForBlock(primFunc, blockName, this.target);
        const sketch = sketches.find(s => s.name === result.sketchName);
        if (sketch) {
          const apply = sketch.instantiate(result.params);
          apply(sch, blockName, this.target);
        }
      }
      return { func: primFunc };
    } catch (e) {
      return null;
    }
  }

  _refineByCostModel(primFunc, blockName, candidates) {
    const topK = candidates.slice(0, this.config.topKForBenchmark);
    const sketches = getSketchesForBlock(primFunc, blockName, this.target);
    const sketchByName = new Map();
    for (const s of sketches) sketchByName.set(s.name, s);

    const measured = [];

    for (const candidate of topK) {
      try {
        const sketch = sketchByName.get(candidate.sketchName);
        if (!sketch) continue;

        const cloned = clonePrimFunc(primFunc);
        const sch = new Schedule(cloned);
        sketch.instantiate(candidate.params)(sch, blockName, this.target);

        const result = this.benchmarkRunner.run(cloned);
        if (!result) continue;

        const score = -result.medianMs;
        measured.push({ ...candidate, score, medianMs: result.medianMs, minMs: result.minMs });

        const features = FeatureExtractor.extract(cloned);
        this.learnedModel.addSample(features, score);
      } catch {
        continue;
      }
    }

    if (measured.length > 0) {
      measured.sort((a, b) => b.score - a.score);
      this.learnedModel.train();
    }

    return measured;
  }

  _computeWorkloadKey(primFunc, blockName) {
    return computeWorkloadKey(primFunc, blockName, this.target);
  }
}
