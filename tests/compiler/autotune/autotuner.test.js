import { describe, it, expect } from 'vitest';
import { getSketchesForBlock, createMultiLevelTileCPUSketch } from '../../../src/compiler/autotune/search_space.js';
import { buildBlockMap } from '../../../src/compiler/autotune/workload_key.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc,
  BlockNode,
  BufferStoreNode,
  BufferLoadNode,
  ForNode,
  VariableNode,
  IntImmNode,
  MathOpNode,
  ForKind,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Schedule } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';
import { CPUTarget, WasmTarget, WebGPUTarget } from '../../../src/backend/target.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { TuningDatabase, TuningRecord, CODEGEN_VERSION } from '../../../src/compiler/autotune/tuning_db.js';
import { RandomSearch, EvolutionarySearch } from '../../../src/compiler/autotune/search.js';
import { Deadline } from '../../../src/compiler/autotune/budget.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { fs } from '../../../src/io/node/fs.js';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function reductionPrimFunc() {
  const out = new Buffer('out', [1], 'f32', 'global');
  const inp = new Buffer('in', [16], 'f32', 'global');
  const load = new BufferLoadNode(inp, []);
  const body = new BufferStoreNode(out, [], load);
  const init = new BufferStoreNode(out, [], load);
  const block = new BlockNode('reduce_blk', [], [{ buffer: inp }], [{ buffer: out }], body, init);
  return new PrimFunc('f', [], block, new Map());
}

describe('getSketchesForBlock classifies reductions from the primFunc', () => {
  it('selects the reduction sketch regardless of whether blockMap is passed', () => {
    const pf = reductionPrimFunc();
    const blockMap = buildBlockMap(pf.body);

    const withMap = getSketchesForBlock(pf, 'reduce_blk', CPUTarget(), blockMap);
    const withoutMap = getSketchesForBlock(pf, 'reduce_blk', CPUTarget());

    expect(withMap[0].name).toBe('reduction_cpu');
    expect(withoutMap[0].name).toBe('reduction_cpu');
  });

  it('picks the GPU reduction sketch (not elementwise) for a reduction block', () => {
    const pf = reductionPrimFunc();
    const sketches = getSketchesForBlock(pf, 'reduce_blk', WebGPUTarget());
    expect(sketches[0].name).toBe('reduction_gpu');
  });
});

describe('Ansor-style sketch auto-generation derives sketches from block structure', () => {
  function matmulPrimFunc(name, M = 8, N = 8, K = 8) {
    const A = new Buffer('A', [M, K], 'float32', 'global');
    const B = new Buffer('B', [K, N], 'float32', 'global');
    const C = new Buffer('C', [M, N], 'float32', 'global');
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

  function elementwisePrimFunc(name, M = 8, N = 8) {
    const A = new Buffer('A', [M, N], 'float32', 'global');
    const C = new Buffer('C', [M, N], 'float32', 'global');
    const m = new VariableNode('m', 'int32');
    const n = new VariableNode('n', 'int32');
    const body = new BufferStoreNode(C, [m, n], new MathOpNode('add', new BufferLoadNode(A, [m, n]), new IntImmNode(1)));
    const iterVars = [{ iterVar: m, binding: m }, { iterVar: n, binding: n }];
    const block = new BlockNode(name, iterVars, [{ buffer: A }], [{ buffer: C }], body, null);
    let nest = block;
    for (const [v, e] of [[n, N], [m, M]]) {
      nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
    }
    return new PrimFunc(name, [], nest, new Map([['A', A], ['C', C]]));
  }

  it('derives multi-level-tiling + reduction sketches for a compute-intensive (reduction+spatial+2-read) block', () => {
    const pf = matmulPrimFunc('gemm');
    expect(getSketchesForBlock(pf, 'gemm', CPUTarget()).map(s => s.name)).toEqual(['mlt_cpu', 'reduction_cpu']);
    expect(getSketchesForBlock(pf, 'gemm', WebGPUTarget()).map(s => s.name)).toEqual(['matmul_gpu', 'reduction_gpu']);
  });

  it('selection is structural, not by block name: a matmul-NAMED elementwise block gets the elementwise sketch', () => {
    const pf = elementwisePrimFunc('matmul_in_name_only');
    expect(getSketchesForBlock(pf, 'matmul_in_name_only', CPUTarget()).map(s => s.name)).toEqual(['elementwise_cpu']);
    expect(getSketchesForBlock(pf, 'matmul_in_name_only', WebGPUTarget()).map(s => s.name)).toEqual(['elementwise_gpu']);
  });

  it('the generated multi-level-tiling sketch produces only valid schedules across its parameter space', () => {
    const sketch = createMultiLevelTileCPUSketch();
    const paramSets = [
      { tile_s0: 4, tile_s1: 4, tile_k: 2 },
      { tile_s0: 2, tile_s1: 1, tile_k: 4 },
      { tile_s0: 8, tile_s1: 8, tile_k: 8 },
      { tile_s0: 1, tile_s1: 1, tile_k: 1 },
      { tile_s0: 256, tile_s1: 128, tile_k: 32 },
    ];
    for (const params of paramSets) {
      const sch = new Schedule(matmulPrimFunc('g'));
      sketch.instantiate(params)(sch, 'g', CPUTarget());
      expect(ScheduleValidator.validate(sch.func), JSON.stringify(params)).toHaveLength(0);
    }
  });
});

describe('autotune end-to-end: tuned output matches baseline on non-power-of-2 shapes (tiling remainders)', () => {
  const F = ScalarType.F32;
  const T = (sh) => new TensorType(sh, F);
  const CASES = [
    { name: 'mm_5x7x3', inTypes: [T([5, 7]), T([7, 3])], build: (b, a) => b.matmul(a[0], a[1]), outShape: [5, 3] },
    { name: 'mm_relu_13x5x11', inTypes: [T([13, 5]), T([5, 11])], build: (b, a) => b.relu(b.matmul(a[0], a[1]).getResult(0)), outShape: [13, 11] },
    { name: 'mm_reduce_7x5', inTypes: [T([7, 5]), T([5, 7])], build: (b, a) => b.reduce(b.matmul(a[0], a[1]).getResult(0), b.scalarConstant(0, F).getResult(0), [1], 'sum'), outShape: [7] },
    { name: 'ew_reduce_9x7', inTypes: [T([9, 7]), T([9, 7])], build: (b, a) => b.reduce(b.tanh(b.mul(a[0], a[1]).getResult(0)).getResult(0), b.scalarConstant(0, F).getResult(0), [0], 'sum'), outShape: [7] },
  ];
  for (const c of CASES) {
    for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`${c.name} autotuned == baseline on ${tname}`, () => {
        const n = c.outShape.reduce((x, y) => x * y, 1);
        const func = buildFunction(c.name, c.inTypes, [T(c.outShape)], (b, a) => { b.returnOp([c.build(b, a).getResult(0)]); });
        const inputs = c.inTypes.map((t) => { const arr = new Float32Array(t.shape.reduce((x, y) => x * y, 1)); for (let i = 0; i < arr.length; i++) arr[i] = Math.sin(i * 1.7) * 1.5; return arr; });
        const r0 = compileGraph(func, makeTarget(), {});
        const ref = new Float32Array(n); r0.run(c.name, ...inputs, ref);
        for (const [strat, seed] of [['random', 7], ['evolutionary', 42]]) {
          const r1 = compileGraph(func, makeTarget(), { scheduling: { enabled: true, autotune: true, strategy: strat, numTrials: 24, populationSize: 12, numGenerations: 4, seed } });
          const out = new Float32Array(n); r1.run(c.name, ...inputs, out);
          for (let i = 0; i < n; i++) expect(Math.abs(ref[i] - out[i]) / (1 + Math.abs(ref[i])), `${c.name}/${tname}/${strat} idx ${i}`).toBeLessThan(2e-3);
        }
      });
    }
  }
});

describe('TuningDatabase serialize/deserialize round-trip + best-score lookup', () => {
  it('preserves records, params, and best-score ordering across serialization', () => {
    const db = new TuningDatabase(1);
    db.store('abc', new TuningRecord('abc', 'matmul_cpu', { tile_m: 64, tile_n: 32, tile_k: 8 }, 12.5, null, 1));
    db.store('abc', new TuningRecord('abc', 'matmul_cpu', { tile_m: 32, tile_n: 16, tile_k: 4 }, 9.0, null, 1));
    db.store('xyz', new TuningRecord('xyz', 'elementwise_cpu', { tile_size: 8, vector_width: 4 }, 5.0, null, 1));

    expect(db.lookup('abc').score).toBe(12.5); // best score returned
    const db2 = TuningDatabase.deserialize(db.serialize());
    expect(db2.size).toBe(db.size);
    expect(db2.lookup('abc').score).toBe(db.lookup('abc').score);
    expect(db2.lookup('abc').params).toEqual(db.lookup('abc').params);
    expect(db2.lookup('xyz').sketchName).toBe('elementwise_cpu');
    expect(db2.has('nope')).toBe(false);
  });
});

describe('TuningDatabase versioning + disk persistence', () => {
  it('invalidates all records when the codegen version no longer matches', () => {
    const db = new TuningDatabase(1);
    db.store('k', new TuningRecord('k', 'matmul_cpu', { tile: 8 }, 3.0, null, 1));
    const blob = db.serialize();
    expect(blob.codegenVersion).toBe(CODEGEN_VERSION);

    const same = TuningDatabase.deserialize(blob);
    expect(same.size).toBe(1);

    const stale = TuningDatabase.deserialize({ ...blob, codegenVersion: 'stale-codegen-xyz' });
    expect(stale.size).toBe(0);
  });

  it('persists to disk and reloads for reuse across sessions', () => {
    const db = new TuningDatabase(1);
    db.store('k', new TuningRecord('k', 'matmul_cpu', { tile_m: 64, tile_k: 8 }, 9.5, null, 1));
    const path = join(tmpdir(), `mlfw-tuningdb-${process.pid}-${db.size}.json`);

    db.saveToFile(path, fs);
    const loaded = TuningDatabase.loadFromFile(path, fs);
    expect(loaded.lookup('k').params).toEqual({ tile_m: 64, tile_k: 8 });
    expect(loaded.lookup('k').score).toBe(9.5);

    fs.remove(path);
    expect(TuningDatabase.loadFromFile(path, fs).size).toBe(0);
  });
});

describe('autotune time budget enforcement', () => {
  const sketch = { name: 's', sampleParams: () => ({}), variables: [] };

  it('RandomSearch stops issuing trials once the deadline is exhausted', () => {
    let now = 0;
    const deadline = new Deadline(50, () => now);
    let calls = 0;
    const evaluator = () => { calls++; now += 20; return { score: 1, features: null }; };
    const search = new RandomSearch({ numTrials: 1000, seed: 1, deadline });
    search.search([sketch], evaluator);
    expect(calls).toBeGreaterThan(0);
    expect(calls).toBeLessThan(10);
  });

  it('EvolutionarySearch stops spawning generations once the deadline is exhausted', () => {
    let now = 0;
    const deadline = new Deadline(30, () => now);
    let calls = 0;
    const evaluator = () => { calls++; now += 10; return { score: 1, features: null }; };
    const search = new EvolutionarySearch({ populationSize: 4, numGenerations: 50, seed: 1, deadline });
    search.search([sketch], evaluator);
    expect(calls).toBeGreaterThan(0);
    expect(calls).toBeLessThan(20);
  });

  it('a zero time budget skips tuning yet still compiles correctly (baseline preserved)', () => {
    const F = ScalarType.F32;
    const T = (sh) => new TensorType(sh, F);
    const func = buildFunction('mm_budget', [T([7, 5]), T([5, 7])], [T([7, 7])], (b, a) => {
      b.returnOp([b.matmul(a[0], a[1]).getResult(0)]);
    });
    const inputs = [T([7, 5]), T([5, 7])].map((t) => {
      const arr = new Float32Array(t.shape.reduce((x, y) => x * y, 1));
      for (let i = 0; i < arr.length; i++) arr[i] = Math.sin(i * 1.3);
      return arr;
    });
    const ref = new Float32Array(49);
    compileGraph(func, CPUTarget(), {}).run('mm_budget', ...inputs, ref);

    const out = new Float32Array(49);
    compileGraph(func, CPUTarget(), { scheduling: { enabled: true, autotune: true, timeBudgetMs: 0 } })
      .run('mm_budget', ...inputs, out);
    for (let i = 0; i < 49; i++) expect(Math.abs(ref[i] - out[i])).toBeLessThan(2e-3);
  });
});

describe('shared TuningDatabase reproduces tuned schedules across sessions', () => {
  const F = ScalarType.F32;
  const T = (sh) => new TensorType(sh, F);
  const mk = () => buildFunction('mm_xsess', [T([6, 4]), T([4, 6])], [T([6, 6])], (b, a) => {
    b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]);
  });
  const inputs = [T([6, 4]), T([4, 6])].map((t) => {
    const arr = new Float32Array(t.shape.reduce((x, y) => x * y, 1));
    for (let i = 0; i < arr.length; i++) arr[i] = Math.cos(i * 0.9) * 1.2;
    return arr;
  });

  it('a cache-hit compile regenerates the same tuned source and stays correct', () => {
    const ref = new Float32Array(36);
    compileGraph(mk(), CPUTarget(), {}).run('mm_xsess', ...inputs, ref);

    const db = new TuningDatabase();
    const sched = { enabled: true, autotune: true, strategy: 'random', numTrials: 8, seed: 1, tuningDB: db };

    const normalize = (src) => src.replace(/_\d+\b/g, '_N');
    const first = compileGraph(mk(), CPUTarget(), { scheduling: { ...sched } });
    expect(db.size).toBeGreaterThan(0);
    const tunedSource = normalize(first.getSource('mm_xsess'));

    const events = [];
    const second = compileGraph(mk(), CPUTarget(), {
      scheduling: { ...sched },
      trace: { level: TraceLevel.VERBOSE, sink: (e) => events.push(e) },
    });

    const autotuneEvents = events.filter((e) => e.type === 'autotune');
    expect(autotuneEvents.some((e) => e.cacheHits > 0)).toBe(true);
    expect(normalize(second.getSource('mm_xsess'))).toBe(tunedSource);

    const out = new Float32Array(36);
    second.run('mm_xsess', ...inputs, out);
    for (let i = 0; i < 36; i++) expect(Math.abs(ref[i] - out[i])).toBeLessThan(2e-3);
  });
});
