import { describe, it, expect } from 'vitest';
import { getSketchesForBlock } from '../../../src/compiler/autotune/search_space.js';
import { buildBlockMap } from '../../../src/compiler/autotune/workload_key.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc,
  BlockNode,
  BufferStoreNode,
  BufferLoadNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { TuningDatabase, TuningRecord } from '../../../src/compiler/autotune/tuning_db.js';

function reductionPrimFunc() {
  const out = new Buffer('out', [1], 'f32', 'global');
  const inp = new Buffer('in', [16], 'f32', 'global');
  const load = new BufferLoadNode(inp, []);
  const body = new BufferStoreNode(out, [], load);
  const init = new BufferStoreNode(out, [], load);
  const block = new BlockNode('reduce_blk', [], [{ buffer: inp }], [{ buffer: out }], body, init);
  return new PrimFunc('f', [], block, new Map());
}

describe('getSketchesForBlock requires blockMap to classify reductions', () => {
  it('selects a different sketch family when blockMap is provided vs omitted', () => {
    const target = CPUTarget();
    const pf = reductionPrimFunc();
    const blockMap = buildBlockMap(pf.body);

    const withMap = getSketchesForBlock(pf, 'reduce_blk', target, blockMap);
    const withoutMap = getSketchesForBlock(pf, 'reduce_blk', target);

    expect(withMap[0].name).not.toBe(withoutMap[0].name);
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
