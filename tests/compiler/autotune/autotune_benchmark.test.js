import { describe, it, beforeEach } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  PrimFunc, ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, BlockRealizeNode, ForKind, MathOpNode, FloatImmNode
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { Autotuner, AutotuneConfig } from '../../../src/compiler/autotune/autotuner.js';
import { BenchmarkRunner } from '../../../src/compiler/autotune/benchmark.js';
import { computeWorkloadKey } from '../../../src/compiler/autotune/workload_key.js';
import { TuningDatabase, TuningRecord } from '../../../src/compiler/autotune/tuning_db.js';

function makeElementwiseFunc(size) {
  const A = new Buffer('A', [size], 'f32', 'global');
  const B = new Buffer('B', [size], 'f32', 'global');
  const C = new Buffer('C', [size], 'f32', 'global');
  const vi = new VariableNode('i', 'int32');
  const viv = new VariableNode('vi', 'int32');
  const bind = new BlockRealizeNode(viv, vi);
  const store = new BufferStoreNode(C, [viv], new MathOpNode('+', new BufferLoadNode(A, [viv]), new BufferLoadNode(B, [viv])));
  const block = new BlockNode('add_block', [bind], [{ buffer: A }, { buffer: B }], [{ buffer: C }], store);
  let body = new ForNode(vi, new IntImmNode(0), new IntImmNode(size), ForKind.SERIAL, block);
  const bufMap = new Map();
  const pA = new VariableNode('pA', 'int32');
  const pB = new VariableNode('pB', 'int32');
  const pC = new VariableNode('pC', 'int32');
  bufMap.set(pA, A);
  bufMap.set(pB, B);
  bufMap.set(pC, C);
  return new PrimFunc('vec_add', [pA, pB, pC], body, bufMap);
}

describe('Structural workload key', () => {
  beforeEach(() => resetVarCounter());

  it('produces deterministic key for same workload', () => {
    const func1 = makeElementwiseFunc(256);
    const func2 = makeElementwiseFunc(256);
    const target = CPUTarget();
    const key1 = computeWorkloadKey(func1, 'add_block', target);
    const key2 = computeWorkloadKey(func2, 'add_block', target);
    assert.equal(key1, key2);
  });

  it('produces different key for different shapes', () => {
    const func1 = makeElementwiseFunc(256);
    const func2 = makeElementwiseFunc(512);
    const target = CPUTarget();
    const key1 = computeWorkloadKey(func1, 'add_block', target);
    const key2 = computeWorkloadKey(func2, 'add_block', target);
    assert.notEqual(key1, key2);
  });

  it('produces different key for different targets', async () => {
    const { GPUTarget } = await import('../../../src/backend/target.js');
    const func = makeElementwiseFunc(256);
    const key1 = computeWorkloadKey(func, 'add_block', CPUTarget());
    const key2 = computeWorkloadKey(func, 'add_block', GPUTarget());
    assert.notEqual(key1, key2);
  });

  it('key is a hex string', () => {
    const func = makeElementwiseFunc(256);
    const key = computeWorkloadKey(func, 'add_block', CPUTarget());
    assert.ok(/^[0-9a-f]{8}$/.test(key), `key should be 8 hex chars, got: ${key}`);
  });
});

describe('BenchmarkRunner', () => {
  beforeEach(() => resetVarCounter());

  it('measures an elementwise kernel', () => {
    const func = makeElementwiseFunc(1024);
    const target = CPUTarget();
    const runner = new BenchmarkRunner(target, { warmup: 2, repeat: 5 });
    const result = runner.run(func);
    assert.ok(result, 'benchmark should produce a result for CPU target');
    assert.ok(result.medianMs >= 0, `medianMs should be non-negative, got ${result.medianMs}`);
    assert.ok(result.minMs >= 0, `minMs should be non-negative, got ${result.minMs}`);
    assert.ok(result.minMs <= result.medianMs, 'min should be <= median');
    assert.ok(result.samples.length >= 5, `should have at least 5 samples, got ${result.samples.length}`);
  });

  it('returns null for invalid primfunc', () => {
    const target = CPUTarget();
    const runner = new BenchmarkRunner(target, { warmup: 1, repeat: 1 });
    const broken = new PrimFunc('broken', [], null, new Map());
    const result = runner.run(broken);
    assert.equal(result, null);
  });
});

describe('Autotuner with benchmark', () => {
  beforeEach(() => resetVarCounter());

  it('autotune with cost model only (no benchmark)', () => {
    const func = makeElementwiseFunc(256);
    const target = CPUTarget();
    const tuner = new Autotuner(target, { strategy: 'random', numTrials: 4, seed: 42, enableBenchmark: false });
    const result = tuner.tuneAndApply(func);
    assert.ok(result);
    assert.ok(result.results.size > 0);
  });

  it('autotune with benchmark enabled', () => {
    const func = makeElementwiseFunc(1024);
    const target = CPUTarget();
    const tuner = new Autotuner(target, {
      strategy: 'random',
      numTrials: 4,
      seed: 42,
      enableBenchmark: true,
      benchmarkWarmup: 1,
      benchmarkRepeat: 3,
      topKForBenchmark: 2,
    });
    const result = tuner.tuneAndApply(func);
    assert.ok(result);
    const blockResult = result.results.get('add_block');
    assert.ok(blockResult, 'should have result for add_block');
  });

  it('uses structural workload key', () => {
    const func = makeElementwiseFunc(256);
    const target = CPUTarget();
    const tuner = new Autotuner(target, { strategy: 'random', numTrials: 4, seed: 42 });
    const key = computeWorkloadKey(func, 'add_block', target);
    assert.ok(/^[0-9a-f]{8}$/.test(key));
  });

  it('caches result and retrieves from DB', () => {
    const func = makeElementwiseFunc(256);
    const target = CPUTarget();
    const tuner = new Autotuner(target, { strategy: 'random', numTrials: 4, seed: 42, useTuningDB: true });

    const result1 = tuner.tune(func);
    assert.ok(result1.size > 0);
    const r1 = result1.get('add_block');
    assert.ok(!r1.fromCache);

    const result2 = tuner.tune(func);
    const r2 = result2.get('add_block');
    assert.ok(r2.fromCache, 'second tune should hit DB cache');
    assert.equal(r1.sketchName, r2.sketchName);
  });

  it('tuning DB serializes and deserializes', () => {
    const db = new TuningDatabase();
    db.store('key1', new TuningRecord('key1', 'sketch1', { a: 1 }, 0.5, null, 1));
    db.store('key1', new TuningRecord('key1', 'sketch1', { a: 2 }, 0.8, null, 1));
    db.store('key2', new TuningRecord('key2', 'sketch2', { b: 3 }, 0.3, null, 1));

    const serialized = db.serialize();
    assert.equal(serialized.entries.length, 3);

    const restored = TuningDatabase.deserialize(serialized);
    assert.ok(restored.has('key1'));
    assert.ok(restored.has('key2'));
    assert.equal(restored.lookup('key1').score, 0.8);
  });
});
