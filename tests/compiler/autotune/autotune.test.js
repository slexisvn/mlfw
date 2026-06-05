import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, BlockRealizeNode, ForKind
} from '../../../src/compiler/ir/tensor/nodes.js';
import { FeatureExtractor, ScheduleFeatures } from '../../../src/compiler/cost_model/features.js';
import { AnalyticalCostModel, LearnedCostModel } from '../../../src/compiler/cost_model/cost_model.js';
import {
  SearchVariable, ScheduleSketch, getSketchesForBlock,
  createElementwiseCPUSketch, createElementwiseGPUSketch
} from '../../../src/compiler/autotune/search_space.js';
import { RandomSearch, EvolutionarySearch } from '../../../src/compiler/autotune/search.js';
import { TuningDatabase, TuningRecord } from '../../../src/compiler/autotune/tuning_db.js';
import { Autotuner, AutotuneConfig } from '../../../src/compiler/autotune/autotuner.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { Schedule } from '../../../src/compiler/schedule/schedule.js';

function makeElemFunc(size) {
  const A = new Buffer('A', [size], 'f32', 'global');
  const B = new Buffer('B', [size], 'f32', 'global');
  const C = new Buffer('C', [size], 'f32', 'global');
  const i = new VariableNode('i', 'int32');
  const vi = new VariableNode('vi', 'int32');
  const bind = new BlockRealizeNode(vi, i);
  const store = new BufferStoreNode(C, [vi], new MathOpNode('+', new BufferLoadNode(A, [vi]), new BufferLoadNode(B, [vi])));
  const block = new BlockNode('add_block', [bind], [{buffer: A}, {buffer: B}], [{buffer: C}], store);
  const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(size), ForKind.SERIAL, block);
  const pA = new VariableNode('pA', 'int32');
  const pB = new VariableNode('pB', 'int32');
  const pC = new VariableNode('pC', 'int32');
  return new PrimFunc('vec_add', [pA, pB, pC], loop, new Map([[pA, A], [pB, B], [pC, C]]));
}

function makeMatmulFunc(M, N, K) {
  const A = new Buffer('A', [M, K], 'f32', 'global');
  const B = new Buffer('B', [K, N], 'f32', 'global');
  const C = new Buffer('C', [M, N], 'f32', 'global');
  const mi = new VariableNode('m', 'int32');
  const ni = new VariableNode('n', 'int32');
  const ki = new VariableNode('k', 'int32');
  const miv = new VariableNode('vm', 'int32');
  const niv = new VariableNode('vn', 'int32');
  const kiv = new VariableNode('vk', 'int32');
  const mB = new BlockRealizeNode(miv, mi);
  const nB = new BlockRealizeNode(niv, ni);
  const kB = new BlockRealizeNode(kiv, ki);
  const initStore = new BufferStoreNode(C, [miv, niv], new FloatImmNode(0));
  const acc = new MathOpNode('+', new BufferLoadNode(C, [miv, niv]), new MathOpNode('*', new BufferLoadNode(A, [miv, kiv]), new BufferLoadNode(B, [kiv, niv])));
  const store = new BufferStoreNode(C, [miv, niv], acc);
  const block = new BlockNode('matmul', [mB, nB, kB], [{buffer: A}, {buffer: B}], [{buffer: C}], store, initStore);
  let body = block;
  body = new ForNode(ki, new IntImmNode(0), new IntImmNode(K), ForKind.SERIAL, body);
  body = new ForNode(ni, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, body);
  body = new ForNode(mi, new IntImmNode(0), new IntImmNode(M), ForKind.SERIAL, body);
  const pA = new VariableNode('pA', 'int32');
  const pB = new VariableNode('pB', 'int32');
  const pC = new VariableNode('pC', 'int32');
  return new PrimFunc('matmul', [pA, pB, pC], body, new Map([[pA, A], [pB, B], [pC, C]]));
}

describe('FeatureExtractor', () => {
  it('extracts features from elementwise', () => {
    const func = makeElemFunc(256);
    const features = FeatureExtractor.extract(func);
    assert.ok(features instanceof ScheduleFeatures);
    assert.equal(features.numLoops, 1);
    assert.equal(features.numBlocks, 1);
    assert.equal(features.totalIterations, 256);
    assert.equal(features.numSerialLoops, 1);
    assert.ok(features.numBufferReads >= 2);
    assert.ok(features.numBufferWrites >= 1);
    assert.ok(features.numMathOps >= 1);
    assert.equal(features.hasReduction, false);
  });

  it('extracts features from matmul', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const features = FeatureExtractor.extract(func);
    assert.equal(features.numLoops, 3);
    assert.equal(features.hasReduction, true);
    assert.ok(features.totalIterations > 0);
    assert.ok(features.numMathOps >= 2);
  });

  it('toVector returns consistent length', () => {
    const f1 = FeatureExtractor.extract(makeElemFunc(64));
    const f2 = FeatureExtractor.extract(makeMatmulFunc(32, 32, 16));
    assert.equal(f1.toVector().length, f2.toVector().length);
    assert.equal(f1.toVector().length, ScheduleFeatures.featureNames().length);
  });

  it('detects thread binding features', () => {
    const func = makeElemFunc(256);
    const sch = new Schedule(func);
    const loops = sch.getLoops('add_block');
    const [bx, tx] = sch.split(loops[0], 32);
    sch.bindThread(bx, 'blockIdx.x');
    sch.bindThread(tx, 'threadIdx.x');
    const features = FeatureExtractor.extract(func);
    assert.equal(features.numThreadBound, 2);
    assert.equal(features.threadBlockSize, 32);
    assert.equal(features.gridSize, 8);
  });
});

describe('AnalyticalCostModel', () => {
  it('scores scheduled better than unscheduled', () => {
    const funcA = makeElemFunc(256);
    const funcB = makeElemFunc(256);
    const sch = new Schedule(funcB);
    const loops = sch.getLoops('add_block');
    const [outer, inner] = sch.split(loops[0], 8);
    sch.parallelize(outer);
    sch.vectorize(inner);
    const model = new AnalyticalCostModel(CPUTarget());
    const scoreA = model.estimate(funcA);
    const scoreB = model.estimate(funcB);
    assert.ok(scoreB.score > scoreA.score);
  });

  it('returns breakdown', () => {
    const model = new AnalyticalCostModel(CPUTarget());
    const est = model.estimate(makeElemFunc(256));
    assert.ok('parallelism' in est.breakdown);
    assert.ok('vectorization' in est.breakdown);
    assert.ok('memoryCoalescing' in est.breakdown);
  });

  it('GPU model scores thread-bound higher', () => {
    const funcA = makeElemFunc(1024);
    const funcB = makeElemFunc(1024);
    const sch = new Schedule(funcB);
    const loops = sch.getLoops('add_block');
    const [bx, tx] = sch.split(loops[0], 128);
    sch.bindThread(bx, 'blockIdx.x');
    sch.bindThread(tx, 'threadIdx.x');
    const model = new AnalyticalCostModel(GPUTarget());
    assert.ok(model.estimate(funcB).score > model.estimate(funcA).score);
  });
});

describe('LearnedCostModel', () => {
  it('trains and predicts', () => {
    const model = new LearnedCostModel();
    const f1 = FeatureExtractor.extract(makeElemFunc(64));
    const f2 = FeatureExtractor.extract(makeElemFunc(256));
    model.addSample(f1, 0.5);
    model.addSample(f2, 0.8);
    model.train();
    assert.ok(model.trained);
    const pred = model.predict(f2);
    assert.ok(typeof pred === 'number');
  });

  it('serializes and deserializes', () => {
    const model = new LearnedCostModel();
    model.addSample(FeatureExtractor.extract(makeElemFunc(64)), 0.5);
    model.train();
    const data = model.serialize();
    const restored = LearnedCostModel.deserialize(data);
    const features = FeatureExtractor.extract(makeElemFunc(64));
    assert.equal(model.predict(features), restored.predict(features));
  });
});

describe('SearchSpace', () => {
  it('SearchVariable samples from candidates', () => {
    const v = new SearchVariable('tile', [4, 8, 16, 32]);
    let idx = 0;
    const rng = (max) => idx++ % max;
    const s1 = v.sample(rng);
    const s2 = v.sample(rng);
    assert.ok([4, 8, 16, 32].includes(s1));
    assert.ok([4, 8, 16, 32].includes(s2));
  });

  it('getSketchesForBlock returns CPU elementwise sketch', () => {
    const func = makeElemFunc(256);
    const sketches = getSketchesForBlock(func, 'add_block', CPUTarget());
    assert.ok(sketches.length > 0);
    assert.equal(sketches[0].name, 'elementwise_cpu');
  });

  it('getSketchesForBlock returns GPU elementwise sketch', () => {
    const func = makeElemFunc(256);
    const sketches = getSketchesForBlock(func, 'add_block', GPUTarget());
    assert.ok(sketches.length > 0);
    assert.equal(sketches[0].name, 'elementwise_gpu');
  });

  it('getSketchesForBlock returns matmul sketch', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const sketches = getSketchesForBlock(func, 'matmul', CPUTarget());
    assert.ok(sketches.length > 0);
    assert.equal(sketches[0].name, 'matmul_cpu');
  });

  it('sketch instantiate applies to schedule', () => {
    const sketch = createElementwiseCPUSketch();
    const params = { tile_size: 16, vector_width: 8 };
    const func = makeElemFunc(256);
    const sch = new Schedule(func);
    const apply = sketch.instantiate(params);
    apply(sch, 'add_block', CPUTarget());
    const loops = sch.getLoops('add_block');
    assert.ok(loops.some(l => l.kind === ForKind.PARALLEL));
    assert.ok(loops.some(l => l.kind === ForKind.VECTORIZED));
  });
});

describe('RandomSearch', () => {
  it('returns sorted candidates', () => {
    const func = makeElemFunc(256);
    const sketches = [createElementwiseCPUSketch()];
    const model = new AnalyticalCostModel(CPUTarget());

    const search = new RandomSearch({ numTrials: 16, seed: 42 });
    const candidates = search.search(sketches, (sketch, params) => {
      try {
        const clone = makeElemFunc(256);
        const sch = new Schedule(clone);
        sketch.instantiate(params)(sch, 'add_block', CPUTarget());
        const features = FeatureExtractor.extract(clone);
        return { score: model.estimateFromFeatures(features).score, features };
      } catch { return null; }
    });

    assert.ok(candidates.length > 0);
    for (let i = 1; i < candidates.length; i++) {
      assert.ok(candidates[i - 1].score >= candidates[i].score);
    }
  });
});

describe('EvolutionarySearch', () => {
  it('returns improved candidates over generations', () => {
    const sketches = [createElementwiseCPUSketch()];
    const model = new AnalyticalCostModel(CPUTarget());

    const search = new EvolutionarySearch({
      populationSize: 16, numGenerations: 5, seed: 123
    });
    const candidates = search.search(sketches, (sketch, params) => {
      try {
        const clone = makeElemFunc(256);
        const sch = new Schedule(clone);
        sketch.instantiate(params)(sch, 'add_block', CPUTarget());
        const features = FeatureExtractor.extract(clone);
        return { score: model.estimateFromFeatures(features).score, features };
      } catch { return null; }
    });

    assert.ok(candidates.length > 0);
    assert.ok(candidates[0].score > 0);
  });
});

describe('TuningDatabase', () => {
  it('stores and retrieves records', () => {
    const db = new TuningDatabase();
    const key = db.computeWorkloadKey('add_block', [256], 'f32', 'cpu_generic');
    const record = new TuningRecord(key, 'elementwise_cpu', { tile: 8 }, 0.85, null, 1);
    db.store(key, record);
    assert.ok(db.has(key));
    const retrieved = db.lookup(key);
    assert.equal(retrieved.score, 0.85);
    assert.equal(retrieved.sketchName, 'elementwise_cpu');
  });

  it('keeps top-k by score', () => {
    const db = new TuningDatabase();
    const key = 'test_key';
    for (let i = 0; i < 15; i++) {
      db.store(key, new TuningRecord(key, 'sketch', {}, i * 0.1, null, 1));
    }
    const top = db.lookupTopK(key, 5);
    assert.equal(top.length, 5);
    assert.ok(top[0].score >= top[1].score);
  });

  it('serializes and deserializes', () => {
    const db = new TuningDatabase();
    const key = 'k1';
    db.store(key, new TuningRecord(key, 's1', { a: 1 }, 0.9, null, 1));
    db.store(key, new TuningRecord(key, 's2', { a: 2 }, 0.7, null, 1));

    const data = db.serialize();
    const restored = TuningDatabase.deserialize(data);
    assert.ok(restored.has(key));
    assert.equal(restored.lookup(key).score, 0.9);
    assert.equal(restored.size, 2);
  });
});

describe('Autotuner', () => {
  it('tunes elementwise block on CPU', () => {
    const func = makeElemFunc(256);
    const tuner = new Autotuner(CPUTarget(), {
      strategy: 'random', numTrials: 8, seed: 42, useTuningDB: true
    });
    const results = tuner.tune(func, 'add_block');
    assert.ok(results.has('add_block'));
    const result = results.get('add_block');
    assert.ok(result.score > 0);
    assert.equal(result.fromCache, false);
  });

  it('returns cached result on second tune', () => {
    const tuner = new Autotuner(CPUTarget(), {
      strategy: 'random', numTrials: 8, seed: 42
    });
    tuner.tune(makeElemFunc(256), 'add_block');
    const results2 = tuner.tune(makeElemFunc(256), 'add_block');
    assert.equal(results2.get('add_block').fromCache, true);
  });

  it('tunes with evolutionary search', () => {
    const func = makeElemFunc(256);
    const tuner = new Autotuner(CPUTarget(), {
      strategy: 'evolutionary', populationSize: 8, numGenerations: 3, seed: 42
    });
    const results = tuner.tune(func, 'add_block');
    assert.ok(results.get('add_block').score > 0);
  });

  it('tuneAndApply modifies the IR', () => {
    const func = makeElemFunc(256);
    const tuner = new Autotuner(CPUTarget(), {
      strategy: 'random', numTrials: 8, seed: 42
    });
    const { func: tuned, applied } = tuner.tuneAndApply(func, 'add_block');
    assert.ok(applied);
  });

  it('tunes matmul block', () => {
    const func = makeMatmulFunc(128, 128, 64);
    const tuner = new Autotuner(CPUTarget(), {
      strategy: 'random', numTrials: 8, seed: 42
    });
    const results = tuner.tune(func, 'matmul');
    assert.ok(results.has('matmul'));
    assert.ok(results.get('matmul').score !== undefined);
  });

  it('tunes all blocks when no blockName specified', () => {
    const func = makeElemFunc(256);
    const tuner = new Autotuner(CPUTarget(), {
      strategy: 'random', numTrials: 4, seed: 42
    });
    const results = tuner.tune(func);
    assert.ok(results.size > 0);
  });

  it('tuning DB persists across autotuner instances', () => {
    const db = new TuningDatabase();
    const tuner1 = new Autotuner(CPUTarget(), { strategy: 'random', numTrials: 8, seed: 42 });
    tuner1.db = db;
    tuner1.tune(makeElemFunc(256), 'add_block');
    assert.ok(db.size > 0);

    const tuner2 = new Autotuner(CPUTarget(), { strategy: 'random', numTrials: 8, seed: 42 });
    tuner2.db = db;
    const results = tuner2.tune(makeElemFunc(256), 'add_block');
    assert.equal(results.get('add_block').fromCache, true);
  });
});
