import { describe, it, expect } from 'vitest';
import { FeatureExtractor, STATEMENT_FEATURE_SCHEMA } from '../../../src/compiler/autotune/features.js';
import { LearnedCostModel, AnalyticalCostModel, GuidedCostModel } from '../../../src/compiler/autotune/cost_model.js';
import { EvolutionarySearch } from '../../../src/compiler/autotune/search.js';
import { TaskScheduler, GradientSchedulerPolicy } from '../../../src/compiler/autotune/task_scheduler.js';
import { Deadline } from '../../../src/compiler/autotune/budget.js';
import { Autotuner } from '../../../src/compiler/autotune/autotuner.js';
import { enumerateFactorizations } from '../../../src/compiler/autotune/factorization.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, BufferLoadNode, ForNode,
  VariableNode, IntImmNode, MathOpNode, ForKind,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { CPUTarget } from '../../../src/backend/target.js';

function matmulPrimFunc(name, M = 8, N = 8, K = 8) {
  const A = new Buffer('A', [M, K], 'f32', 'global');
  const B = new Buffer('B', [K, N], 'f32', 'global');
  const C = new Buffer('C', [M, N], 'f32', 'global');
  const m = new VariableNode('m', 'int32');
  const n = new VariableNode('n', 'int32');
  const k = new VariableNode('k', 'int32');
  const prod = new MathOpNode('mul', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n]));
  const acc = new MathOpNode('add', new BufferLoadNode(C, [m, n]), prod);
  const body = new BufferStoreNode(C, [m, n], acc);
  const init = new BufferStoreNode(C, [m, n], new IntImmNode(0));
  const iterVars = [{ iterVar: m, binding: m }, { iterVar: n, binding: n }, { iterVar: k, binding: k }];
  const block = new BlockNode(name, iterVars, [{ buffer: A }, { buffer: B }], [{ buffer: C }], body, init);
  let nest = block;
  for (const [v, e] of [[k, K], [n, N], [m, M]]) {
    nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  }
  return new PrimFunc(name, [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

const sidx = (name) => STATEMENT_FEATURE_SCHEMA.indexOf(name);

describe('Ansor multi-level tile factorization', () => {
  it('every factor-tuple multiplies back to the extent and has the requested level count', () => {
    for (const [extent, levels] of [[8, 3], [64, 4], [256, 4], [96, 2], [12, 3]]) {
      const tuples = enumerateFactorizations(extent, levels);
      expect(tuples.length).toBeGreaterThan(0);
      for (const t of tuples) {
        expect(t.length).toBe(levels);
        expect(t.reduce((a, b) => a * b, 1)).toBe(extent);
      }
    }
  });

  it('includes a degenerate tuple so prime extents still tile validly', () => {
    for (const [extent, levels] of [[7, 4], [13, 3], [5, 4]]) {
      const tuples = enumerateFactorizations(extent, levels);
      const degenerate = tuples.some(t => t.filter(x => x === 1).length === levels - 1 && t.includes(extent));
      expect(degenerate, `extent ${extent}`).toBe(true);
    }
  });

  it('is bounded and deterministic for a fixed extent/levels', () => {
    const a = enumerateFactorizations(256, 4);
    const b = enumerateFactorizations(256, 4);
    expect(a.length).toBeLessThanOrEqual(48);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});

describe('Ansor per-statement feature extraction', () => {
  it('emits one fixed-width vector per compute statement with per-access features', () => {
    const pf = matmulPrimFunc('gemm', 8, 8, 8);
    const vecs = FeatureExtractor.extractStatements(pf);

    expect(vecs.length).toBe(2);
    for (const v of vecs) expect(v.length).toBe(STATEMENT_FEATURE_SCHEMA.length);

    const reads = vecs.map(v => v[sidx('numReads')]);
    expect(Math.max(...reads)).toBe(3);

    const iters = vecs.map(v => v[sidx('iterCount')]);
    expect(Math.max(...iters)).toBe(8 * 8 * 8);

    for (const v of vecs) expect(v[sidx('underReduction')]).toBe(1);
  });
});

describe('EvolutionarySearch crossover', () => {
  const mkVar = (name, cands) => ({ name, candidates: cands, sample: (rng) => cands[rng(cands.length)] });
  const sketch = {
    name: 's',
    variables: [mkVar('a', [1, 2, 3, 4]), mkVar('b', [10, 20, 30])],
    sampleParams(rng) { const p = {}; for (const v of this.variables) p[v.name] = v.sample(rng); return p; },
  };

  it('produces children whose every gene comes from one of the two parents', () => {
    const es = new EvolutionarySearch({ seed: 5 });
    const pa = { sketch, params: { a: 1, b: 10 } };
    const pb = { sketch, params: { a: 4, b: 30 } };
    for (let i = 0; i < 20; i++) {
      const child = es._crossover(pa, pb);
      expect([1, 4]).toContain(child.a);
      expect([10, 30]).toContain(child.b);
    }
  });

  it('searches a valid, deterministic space and improves toward the optimum', () => {
    const evaluator = (sk, params) => ({ score: params.a + params.b });
    const run = () => new EvolutionarySearch({ populationSize: 8, numGenerations: 6, seed: 1 })
      .search([sketch], evaluator);

    const r1 = run();
    const r2 = run();
    expect(r1.candidates.length).toBeGreaterThan(0);
    for (const c of r1.candidates) {
      expect([1, 2, 3, 4]).toContain(c.params.a);
      expect([10, 20, 30]).toContain(c.params.b);
    }
    expect(r1.candidates[0].params).toEqual(r2.candidates[0].params);
    expect(r1.candidates[0].score).toBe(34);
  });
});

describe('LearnedCostModel per-statement sum formulation', () => {
  it('learns a sum-of-statements target and predicts monotonically', () => {
    const m = new LearnedCostModel(null, { seed: 2, epochs: 1500 });
    for (let i = 0; i < 40; i++) {
      const k = 1 + (i % 3);
      const stmts = [];
      let y = 0;
      for (let j = 0; j < k; j++) {
        const x = (i * 7 + j * 3) % 10;
        stmts.push([x, 1]);
        y += x;
      }
      m.addSample(stmts, y);
    }
    m.train();

    expect(m.trained).toBe(true);
    expect(m.predict([])).toBe(0);
    const low = m.predict([[1, 1]]);
    const high = m.predict([[9, 1], [9, 1]]);
    expect(high).toBeGreaterThan(low);
  });

  it('ignores empty statement sets', () => {
    const m = new LearnedCostModel(null, { seed: 1 });
    m.addSample([], 5);
    expect(m._X.length).toBe(0);
  });
});

describe('GuidedCostModel closes the loop: learned model steers once trained', () => {
  it('falls back to analytical cold, keeps analytical until confident, then routes to learned.predict', () => {
    const analytical = new AnalyticalCostModel(CPUTarget());
    const learned = new LearnedCostModel(null, { seed: 4 });
    const guided = new GuidedCostModel(analytical, learned, { confidenceSamples: 8 });
    const pf = matmulPrimFunc('gemm', 8, 8, 8);

    expect(guided.score(pf)).toBe(analytical.score(pf));

    const dim = STATEMENT_FEATURE_SCHEMA.length;
    for (let i = 0; i < 6; i++) learned.addSample([new Array(dim).fill(i)], i * 1.5);
    learned.train();

    expect(learned.trained).toBe(true);
    expect(guided.score(pf)).toBe(analytical.score(pf));

    for (let i = 6; i < 8; i++) learned.addSample([new Array(dim).fill(i)], i * 1.5);
    learned.train();

    expect(guided.score(pf)).toBe(learned.predict(FeatureExtractor.extractStatements(pf)));
  });
});

describe('TaskScheduler allocates budget by weight and respects the deadline', () => {
  it('gives more rounds to the higher-weight, still-improving task', () => {
    let t = 0;
    const clock = () => t;
    const mkTask = (weight, gain) => ({ weight, session: { plateaued: false, runRound() { t += 1; return gain; } } });
    const high = mkTask(10, 1);
    const low = mkTask(1, 1);

    const sched = new TaskScheduler(new GradientSchedulerPolicy());
    sched.run([high, low], new Deadline(12, clock), { maxRoundsPerTask: 100, plateauPatience: 5 });

    expect(high.rounds).toBeGreaterThan(low.rounds);
    expect(high.rounds + low.rounds).toBeLessThanOrEqual(12);
  });

  it('stops scheduling a task that reports itself plateaued after one round', () => {
    const task = { weight: 1, session: { plateaued: false, runRound() { this.plateaued = true; return 0; } } };
    const sched = new TaskScheduler();
    sched.run([task], new Deadline(Infinity), {});
    expect(task.rounds).toBe(1);
    expect(task.plateaued).toBe(true);
  });
});

describe('Autotuner online loop trains the cost model from measurements', () => {
  it('measures top-K, trains the learned model, and returns a measured best', () => {
    const autotuner = new Autotuner(CPUTarget(), {
      strategy: 'evolutionary', populationSize: 6, numGenerations: 2, seed: 3,
      timeBudgetMs: 60000, topKForBenchmark: 3, maxRoundsPerTask: 3, useTuningDB: false,
    });
    autotuner.benchmarkRunner = {
      run(primFunc) {
        const vecs = FeatureExtractor.extractStatements(primFunc);
        let work = 0;
        for (const v of vecs) work += v[sidx('iterCount')] || 0;
        const medianMs = 1 + (work % 7);
        return { medianMs, minMs: medianMs };
      },
    };

    const pf = matmulPrimFunc('gemm', 8, 8, 8);
    const results = autotuner.tune(pf, 'gemm');

    const r = results.get('gemm');
    expect(r).toBeDefined();
    expect(autotuner.learnedModel.trained).toBe(true);
    expect(typeof r.medianMs).toBe('number');
    expect(r.medianMs).toBeGreaterThanOrEqual(0);
  });
});
