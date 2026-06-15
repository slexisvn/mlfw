import { Schedule } from '../schedule/schedule.js';
import { ScheduleValidator } from '../schedule/validator.js';
import { FeatureExtractor } from './features.js';
import { AnalyticalCostModel, LearnedCostModel } from './cost_model.js';
import { getSketchesForBlock } from './search_space.js';
import { RandomSearch, EvolutionarySearch } from './search.js';
import { TuningRecord, TuningDatabase } from './tuning_db.js';
import { BenchmarkRunner } from './benchmark.js';
import { Deadline } from './budget.js';
import { buildBlockMap, computeWorkloadKey } from './workload_key.js';
import { PrimFunc, ForNode, SeqNode, BlockNode } from '../ir/tensor/nodes.js';
import { getMeasurer } from '../runtime/measurer_registry.js';

function resolveMeasurer(target) {
  if (target.isCPU()) return null;
  const measurer = getMeasurer(target.kind);
  if (!measurer) {
    throw new Error('hardwareMeasure requested for target \'' + target.kind + '\' but no measurer is registered for it; the corresponding runtime must be loaded (Node: import \'#io/cuda_runtime\') before compiling');
  }
  return measurer;
}

class AutotuneConfig {
  constructor(opts = {}) {
    this.strategy = opts.strategy || 'evolutionary';
    this.numTrials = opts.numTrials || 64;
    this.populationSize = opts.populationSize || 32;
    this.numGenerations = opts.numGenerations || 10;
    this.seed = opts.seed || 42;
    this.timeBudgetMs = opts.timeBudgetMs || 30000;
    this.clock = opts.clock || null;
    this.tuningDB = opts.tuningDB || null;
    this.useTuningDB = opts.useTuningDB !== false;
    this.measurer = opts.measurer || null;
    this.hardwareMeasure = opts.hardwareMeasure ?? false;
    this.enableBenchmark = opts.enableBenchmark ?? (this.hardwareMeasure || !!opts.measurer);
    this.benchmarkWarmup = opts.benchmarkWarmup ?? 3;
    this.benchmarkRepeat = opts.benchmarkRepeat ?? 10;
    this.benchmarkMaxCv = opts.benchmarkMaxCv ?? 0;
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
    copy.type = node.type;
    copy._parent = null;
    copy._parentKey = null;
    copy._parentIdx = -1;
    switch (node.type) {
      case 'PrimFunc':
        copy.name = node.name;
        copy.params = node.params;
        copy.body = cloneNode(node.body);
        copy.bufferMap = new Map(node.bufferMap);
        copy.shapeParams = node.shapeParams;
        copy._setChild('body', copy.body);
        break;
      case 'ForNode':
        copy.loopVar = node.loopVar;
        copy.min = cloneNode(node.min);
        copy.extent = cloneNode(node.extent);
        copy.kind = node.kind;
        copy.body = cloneNode(node.body);
        copy.threadTag = node.threadTag;
        copy._setChild('body', copy.body);
        break;
      case 'BlockNode':
        copy.name = node.name;
        copy.iterVars = node.iterVars.map(cloneNode);
        copy.reads = node.reads;
        copy.writes = node.writes;
        copy.body = cloneNode(node.body);
        copy.initBody = cloneNode(node.initBody);
        copy._setChild('body', copy.body);
        copy._setChild('initBody', copy.initBody);
        break;
      case 'SeqNode':
        copy.stmts = node.stmts.map(cloneNode);
        copy._setChildren('stmts', copy.stmts);
        break;
      case 'AllocateNode':
        copy.buffer = node.buffer;
        copy.scope = node.scope;
        copy.body = cloneNode(node.body);
        copy._setChild('body', copy.body);
        break;
      case 'LetStmtNode':
        copy.variable = node.variable;
        copy.value = cloneNode(node.value);
        copy.body = cloneNode(node.body);
        copy._setChild('body', copy.body);
        break;
      case 'IfThenElseNode':
        copy.condition = cloneNode(node.condition);
        copy.thenBody = cloneNode(node.thenBody);
        copy.elseBody = cloneNode(node.elseBody);
        copy._setChild('thenBody', copy.thenBody);
        copy._setChild('elseBody', copy.elseBody);
        break;
      case 'WhileNode':
        copy.condVar = node.condVar;
        copy.condBody = cloneNode(node.condBody);
        copy.loopBody = cloneNode(node.loopBody);
        copy._setChild('condBody', copy.condBody);
        copy._setChild('loopBody', copy.loopBody);
        break;
      default:
        for (const key of Object.keys(node)) {
          if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
          const val = node[key];
          if (val instanceof Map) copy[key] = new Map(val);
          else if (Array.isArray(val)) copy[key] = val.map(cloneNode);
          else if (typeof val === 'object' && val !== null && val.type) copy[key] = cloneNode(val);
          else copy[key] = val;
        }
        break;
    }
    return copy;
  };
  return cloneNode(primFunc);
}

function extractBlockMini(primFunc, blockName, blockMap) {
  const block = blockMap.get(blockName);
  if (!block) return null;

  const path = [];
  let cur = block._parent;
  while (cur && cur !== primFunc) {
    if (cur.type === 'ForNode') path.push(cur);
    cur = cur._parent;
  }
  path.reverse();

  let body = cloneBlockSubtree(block);

  for (let i = path.length - 1; i >= 0; i--) {
    const loop = path[i];
    const wrapper = new ForNode(
      loop.loopVar,
      cloneBlockSubtree(loop.min),
      cloneBlockSubtree(loop.extent),
      loop.kind,
      body,
      loop.threadTag
    );
    wrapper._setChild('body', body);
    body = wrapper;
  }

  const bufs = new Map();
  for (const r of block.reads) bufs.set(r.buffer.name, r.buffer);
  for (const w of block.writes) bufs.set(w.buffer.name, w.buffer);

  const params = [];
  for (const p of primFunc.params) {
    if (bufs.has(p.name)) params.push(p);
  }

  return new PrimFunc('__tune_' + blockName, params, body, bufs, []);
}

function cloneBlockSubtree(block) {
  const cloneNode = (node) => {
    if (!node || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(cloneNode);
    const copy = Object.create(Object.getPrototypeOf(node));
    copy.type = node.type;
    copy._parent = null;
    copy._parentKey = null;
    copy._parentIdx = -1;
    for (const key of Object.keys(node)) {
      if (key === '_parent' || key === '_parentKey' || key === '_parentIdx') continue;
      const val = node[key];
      if (val instanceof Map) copy[key] = new Map(val);
      else if (Array.isArray(val)) copy[key] = val.map(cloneNode);
      else if (typeof val === 'object' && val !== null && val.type) copy[key] = cloneNode(val);
      else copy[key] = val;
    }
    if (copy._setChild) {
      if (copy.body) copy._setChild('body', copy.body);
      if (copy.initBody) copy._setChild('initBody', copy.initBody);
      if (copy.thenBody) copy._setChild('thenBody', copy.thenBody);
      if (copy.elseBody) copy._setChild('elseBody', copy.elseBody);
    }
    if (copy._setChildren && copy.stmts) copy._setChildren('stmts', copy.stmts);
    return copy;
  };
  return cloneNode(block);
}

export class Autotuner {
  constructor(target, config = {}) {
    this.target = target;
    this.config = config instanceof AutotuneConfig ? config : new AutotuneConfig(config);
    if (this.config.hardwareMeasure) this.config.measurer = resolveMeasurer(target);
    this.costModel = new AnalyticalCostModel(target);
    this.learnedModel = new LearnedCostModel();
    this.db = this.config.tuningDB instanceof TuningDatabase ? this.config.tuningDB : new TuningDatabase();
    this.benchmarkRunner = this.config.enableBenchmark
      ? new BenchmarkRunner(target, {
          warmup: this.config.benchmarkWarmup,
          repeat: this.config.benchmarkRepeat,
          maxCv: this.config.benchmarkMaxCv,
          measurer: this.config.measurer
        })
      : null;
  }

  tune(primFunc, blockName = null) {
    const blockNames = blockName ? [blockName] : collectBlockNames(primFunc.body);
    const blockMap = buildBlockMap(primFunc.body);
    const results = new Map();
    const keyToResult = new Map();
    const deadline = new Deadline(this.config.timeBudgetMs, this.config.clock);

    for (const name of blockNames) {
      const key = computeWorkloadKey(primFunc, name, this.target, blockMap);
      if (keyToResult.has(key)) {
        results.set(name, keyToResult.get(key));
        continue;
      }
      const result = this._tuneBlock(primFunc, name, blockMap, deadline);
      if (result) {
        results.set(name, result);
        keyToResult.set(key, result);
      }
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

  _tuneBlock(primFunc, blockName, blockMap, deadline = null) {
    const workloadKey = computeWorkloadKey(primFunc, blockName, this.target, blockMap);

    if (this.config.useTuningDB && this.db.has(workloadKey)) {
      const cached = this.db.lookup(workloadKey);
      return { sketchName: cached.sketchName, params: cached.params, score: cached.score, fromCache: true };
    }

    const richGpu = !!this.config.measurer;
    const sketches = getSketchesForBlock(primFunc, blockName, this.target, blockMap, { richGpu });
    if (sketches.length === 0) return null;

    if (this.benchmarkRunner && sketches.length === 1 && typeof sketches[0].enumerate === 'function') {
      const enumerated = sketches[0].enumerate().map(params => ({ sketchName: sketches[0].name, params }));
      const measured = this._refineByCostModel(primFunc, blockName, enumerated, blockMap, deadline, enumerated.length);
      if (measured.length === 0) return null;
      const best = measured[0];
      if (this.config.useTuningDB) {
        const record = new TuningRecord(workloadKey, best.sketchName, best.params, best.score, null, this.db.version);
        record.medianMs = best.medianMs || null;
        record.minMs = best.minMs || null;
        this.db.store(workloadKey, record);
      }
      return { sketchName: best.sketchName, params: best.params, score: best.score, fromCache: false, medianMs: best.medianMs, minMs: best.minMs };
    }

    const mini = extractBlockMini(primFunc, blockName, blockMap);
    const miniBlockName = mini ? blockName : null;
    const evalTarget = mini || primFunc;

    const evaluator = (sketch, params) => {
      return this._evaluate(evalTarget, miniBlockName || blockName, sketch, params);
    };

    let candidates;
    if (this.config.strategy === 'evolutionary') {
      const search = new EvolutionarySearch({
        populationSize: this.config.populationSize,
        numGenerations: this.config.numGenerations,
        seed: this.config.seed,
        deadline
      });
      candidates = search.search(sketches, evaluator);
    } else {
      const search = new RandomSearch({
        numTrials: this.config.numTrials,
        seed: this.config.seed,
        deadline
      });
      candidates = search.search(sketches, evaluator);
    }

    if (candidates.length === 0) return null;

    let best = candidates[0];

    if (this.benchmarkRunner && candidates.length > 0) {
      const measured = this._refineByCostModel(primFunc, blockName, candidates, blockMap, deadline);
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
      const blockMap = buildBlockMap(primFunc.body);
      const applied = new Set();
      for (const [blockName, result] of tuneResults) {
        if (applied.has(result)) continue;
        applied.add(result);
        if (!result.sketchName || !result.params) continue;
        const sketches = getSketchesForBlock(primFunc, blockName, this.target, blockMap, { richGpu: !!this.config.measurer });
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

  _refineByCostModel(primFunc, blockName, candidates, blockMap, deadline = null, maxK = null) {
    const topK = candidates.slice(0, maxK ?? this.config.topKForBenchmark);
    const sketches = getSketchesForBlock(primFunc, blockName, this.target, blockMap, { richGpu: !!this.config.measurer });
    const sketchByName = new Map();
    for (const s of sketches) sketchByName.set(s.name, s);

    const measured = [];

    for (const candidate of topK) {
      if (deadline && deadline.expired) break;
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

}
