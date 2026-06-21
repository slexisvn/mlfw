import { Schedule } from '../schedule/schedule.js';
import { AnalyticalCostModel, LearnedCostModel, GuidedCostModel } from './cost_model.js';
import { getSketchesForBlock } from './search_space.js';
import { TuningRecord, TuningDatabase } from './tuning_db.js';
import { BenchmarkRunner } from './benchmark.js';
import { Deadline } from './budget.js';
import { buildBlockMap, computeWorkloadKey } from './workload_key.js';
import { collectAllBlockNames } from './block_analysis.js';
import { buildBlockDAG, findFusibleConsumer } from './block_dag.js';
import { classifyBlock } from '../schedule/rules.js';
import { BlockTuningSession, gpuThreadBlockSize } from './session.js';
import { clonePrimFunc } from './tune_ir.js';
import { TaskScheduler } from './task_scheduler.js';
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
    this.mutationRate = opts.mutationRate;
    this.eliteRatio = opts.eliteRatio;
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
    this.maxRoundsPerTask = opts.maxRoundsPerTask ?? 8;
    this.plateauPatience = opts.plateauPatience ?? 2;
    this.schedulerPolicy = opts.schedulerPolicy || null;
  }
}

export class Autotuner {
  constructor(target, config = {}) {
    this.target = target;
    this.config = config instanceof AutotuneConfig ? config : new AutotuneConfig(config);
    if (this.config.hardwareMeasure) this.config.measurer = resolveMeasurer(target);
    this.analyticalModel = new AnalyticalCostModel(target);
    this.learnedModel = new LearnedCostModel();
    this.costModel = new GuidedCostModel(this.analyticalModel, this.learnedModel);
    this.db = this.config.tuningDB instanceof TuningDatabase ? this.config.tuningDB : new TuningDatabase();
    this.benchmarkRunner = this.config.enableBenchmark
      ? new BenchmarkRunner(target, {
          warmup: this.config.benchmarkWarmup,
          repeat: this.config.benchmarkRepeat,
          maxCv: this.config.benchmarkMaxCv,
          measurer: this.config.measurer
        })
      : null;
    this.scheduler = new TaskScheduler(this.config.schedulerPolicy);
  }

  tune(primFunc, blockName = null) {
    const blockNames = blockName ? [blockName] : collectAllBlockNames(primFunc.body);
    const blockMap = buildBlockMap(primFunc.body);
    const deadline = new Deadline(this.config.timeBudgetMs, this.config.clock);

    const tasksByKey = new Map();
    const keyByBlock = new Map();

    for (const name of blockNames) {
      const key = computeWorkloadKey(primFunc, name, this.target, blockMap);
      keyByBlock.set(name, key);
      const existing = tasksByKey.get(key);
      if (existing) { existing.weight++; continue; }

      if (this.config.useTuningDB && this.db.has(key)) {
        tasksByKey.set(key, { key, kind: 'cache', cached: this.db.lookup(key), weight: 1 });
        continue;
      }

      const sketches = getSketchesForBlock(primFunc, name, this.target, blockMap, { richGpu: !!this.config.measurer });
      if (sketches.length === 0) {
        tasksByKey.set(key, { key, kind: 'empty', weight: 1 });
        continue;
      }

      const session = new BlockTuningSession({
        target: this.target, primFunc, blockName: name, blockMap, sketches,
        costModel: this.costModel, learnedModel: this.learnedModel,
        benchmarkRunner: this.benchmarkRunner, config: this.config, deadline
      });
      tasksByKey.set(key, { key, kind: 'session', session, weight: 1 });
    }

    const sessionTasks = [...tasksByKey.values()].filter(t => t.kind === 'session');
    if (sessionTasks.length > 0) this.scheduler.run(sessionTasks, deadline, this.config);

    const results = new Map();
    for (const name of blockNames) {
      const task = tasksByKey.get(keyByBlock.get(name));
      if (task.kind === 'cache') {
        results.set(name, { sketchName: task.cached.sketchName, params: task.cached.params, score: task.cached.score, fromCache: true });
        continue;
      }
      if (task.kind === 'empty') continue;

      const best = task.session.best();
      if (!best) continue;

      if (this.config.useTuningDB && !task.stored) {
        const record = new TuningRecord(task.key, best.sketchName, best.params, best.score, null, this.db.version);
        record.medianMs = best.medianMs || null;
        record.minMs = best.minMs || null;
        this.db.store(task.key, record);
        task.stored = true;
      }

      results.set(name, { sketchName: best.sketchName, params: best.params, score: best.score, fromCache: false, medianMs: best.medianMs, minMs: best.minMs });
    }

    return results;
  }

  tuneAndApply(primFunc, blockName = null) {
    const tuneResults = this.tune(primFunc, blockName);
    if (tuneResults.size > 0) {
      const best = this._applyBestSchedule(primFunc, tuneResults);
      if (best) return { func: best.func, results: tuneResults, applied: true };
    }
    return { func: primFunc, results: tuneResults, applied: false };
  }

  _applyBestSchedule(primFunc, tuneResults) {
    try {
      const sch = new Schedule(primFunc);
      const blockMap = buildBlockMap(primFunc.body);
      const dag = buildBlockDAG(primFunc);

      const fusedAway = new Set();
      const ordered = [];
      for (const entry of tuneResults) {
        if (entry[1].sketchName === 'fused') {
          const consumer = findFusibleConsumer(primFunc, dag, entry[0], classifyBlock);
          if (consumer) fusedAway.add(consumer);
          ordered.unshift(entry);
        } else {
          ordered.push(entry);
        }
      }

      const applied = new Set();
      for (const [blockName, result] of ordered) {
        if (fusedAway.has(blockName) || applied.has(result)) continue;
        applied.add(result);
        if (!result.sketchName || !result.params) continue;
        try {
          const sketches = getSketchesForBlock(primFunc, blockName, this.target, blockMap, { richGpu: !!this.config.measurer });
          const sketch = sketches.find(s => s.name === result.sketchName);
          if (sketch && this._fitsThreadBlock(primFunc, blockName, sketch, result.params)) {
            const apply = sketch.instantiate(result.params);
            apply(sch, blockName, this.target);
          }
        } catch (e) {
          continue;
        }
      }
      return { func: primFunc };
    } catch (e) {
      return null;
    }
  }

  _fitsThreadBlock(primFunc, blockName, sketch, params) {
    if (!this.target.isGPU || !this.target.isGPU() || !this.target.maxThreadsPerBlock) return true;
    try {
      const trial = clonePrimFunc(primFunc);
      sketch.instantiate(params)(new Schedule(trial), blockName, this.target);
      return gpuThreadBlockSize(trial) <= this.target.maxThreadsPerBlock;
    } catch (e) {
      return false;
    }
  }
}
