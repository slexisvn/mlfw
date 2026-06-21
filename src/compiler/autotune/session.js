import { Schedule } from '../schedule/schedule.js';
import { ScheduleValidator } from '../schedule/validator.js';
import { FeatureExtractor } from './features.js';
import { clonePrimFunc, extractBlockMini } from './tune_ir.js';
import { createSearchStrategy } from './search.js';
import { LearnedCostModel, GuidedCostModel } from './cost_model.js';

const THREAD_AXES = ['threadIdx.x', 'threadIdx.y', 'threadIdx.z'];

export function gpuThreadBlockSize(func) {
  const perAxis = { 'threadIdx.x': 1, 'threadIdx.y': 1, 'threadIdx.z': 1 };
  const seen = new Set();
  const visit = (node) => {
    if (!node || typeof node !== 'object' || seen.has(node)) return;
    seen.add(node);
    if (node.type === 'ForNode' && perAxis[node.threadTag] !== undefined) {
      const extent = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : 1;
      if (extent > perAxis[node.threadTag]) perAxis[node.threadTag] = extent;
    }
    for (const key in node) {
      const child = node[key];
      if (Array.isArray(child)) child.forEach(visit);
      else if (child && typeof child === 'object') visit(child);
    }
  };
  visit(func.body);
  return THREAD_AXES.reduce((product, axis) => product * perAxis[axis], 1);
}

export class BlockTuningSession {
  constructor(opts) {
    this.target = opts.target;
    this.primFunc = opts.primFunc;
    this.blockName = opts.blockName;
    this.sketches = opts.sketches;
    this.benchmarkRunner = opts.benchmarkRunner || null;
    this.config = opts.config;
    this.deadline = opts.deadline || null;

    const needsWholeFunc = this.sketches.some(s => s.name === 'fused');
    if (needsWholeFunc) {
      this.learnedModel = new LearnedCostModel();
      this.costModel = new GuidedCostModel(opts.costModel.analytical, this.learnedModel);
    } else {
      this.costModel = opts.costModel;
      this.learnedModel = opts.learnedModel;
    }
    const mini = needsWholeFunc ? null : extractBlockMini(opts.primFunc, opts.blockName, opts.blockMap);
    this.evalFunc = mini || opts.primFunc;
    this.evalBlockName = opts.blockName;

    this.sketchByName = new Map();
    for (const s of this.sketches) this.sketchByName.set(s.name, s);

    const single = this.sketches.length === 1 && typeof this.sketches[0].enumerate === 'function';
    this.enumSketch = single ? this.sketches[0] : null;
    this.enumParams = single ? this.sketches[0].enumerate() : null;

    this.strategy = createSearchStrategy({ ...this.config, deadline: this.deadline });
    this.population = null;
    this._best = null;
    this.plateaued = false;
  }

  runRound() {
    const prev = this._best ? this._best.measuredScore : -Infinity;
    const candidates = this._produceCandidates();
    if (candidates.length === 0) {
      this.plateaued = true;
      return 0;
    }
    if (this.benchmarkRunner) {
      this._measureAndLearn(candidates);
    } else {
      const top = candidates[0];
      this._consider({ sketchName: top.sketchName, params: top.params, score: top.score, measuredScore: top.score });
      this.plateaued = true;
    }
    const now = this._best ? this._best.measuredScore : -Infinity;
    return Math.max(0, now - prev);
  }

  best() {
    if (!this._best) return null;
    return {
      sketchName: this._best.sketchName,
      params: this._best.params,
      score: this._best.score,
      medianMs: this._best.medianMs ?? null,
      minMs: this._best.minMs ?? null
    };
  }

  _produceCandidates() {
    if (this.enumSketch) {
      const scored = [];
      for (const params of this.enumParams) {
        const r = this._evaluate(this.enumSketch, params);
        if (r) scored.push({ sketchName: this.enumSketch.name, params, score: r.score });
      }
      scored.sort((a, b) => b.score - a.score);
      return scored;
    }
    const evaluator = (sketch, params) => this._evaluate(sketch, params);
    const { candidates, population } = this.strategy.search(this.sketches, evaluator, this.population);
    this.population = population;
    return candidates;
  }

  _evaluate(sketch, params) {
    try {
      const cloned = clonePrimFunc(this.evalFunc);
      const sch = new Schedule(cloned);
      sketch.instantiate(params)(sch, this.evalBlockName, this.target);
      const errors = ScheduleValidator.validate(cloned);
      if (errors.length > 0) return null;
      const blockLimit = this.target.maxThreadsPerBlock;
      if (this.target.isGPU && this.target.isGPU() && blockLimit && gpuThreadBlockSize(cloned) > blockLimit) return null;
      return { score: this.costModel.score(cloned) };
    } catch (e) {
      return null;
    }
  }

  _measureAndLearn(candidates) {
    const topK = candidates.slice(0, this.config.topKForBenchmark);
    for (const cand of topK) {
      if (this.deadline && this.deadline.expired) break;
      const measured = this._measure(cand);
      if (!measured) continue;
      const score = -measured.result.medianMs;
      this.learnedModel.addSample(measured.features, score);
      this._consider({
        sketchName: cand.sketchName, params: cand.params, score,
        measuredScore: score, medianMs: measured.result.medianMs, minMs: measured.result.minMs
      });
    }
    this.learnedModel.train();
  }

  _measure(cand) {
    const sketch = this.sketchByName.get(cand.sketchName);
    if (!sketch) return null;
    let scheduled, miniScheduled;
    try {
      scheduled = clonePrimFunc(this.primFunc);
      sketch.instantiate(cand.params)(new Schedule(scheduled), this.blockName, this.target);
      miniScheduled = clonePrimFunc(this.evalFunc);
      sketch.instantiate(cand.params)(new Schedule(miniScheduled), this.evalBlockName, this.target);
    } catch (e) {
      return null;
    }
    const result = this.benchmarkRunner.run(scheduled);
    if (!result) return null;
    return { result, features: FeatureExtractor.extractStatements(miniScheduled) };
  }

  _consider(rec) {
    if (!this._best || rec.measuredScore > this._best.measuredScore) this._best = rec;
  }
}
