import { describe, it, expect } from 'vitest';
import { spatialIter, reduceIter } from '../../_utils/ir_fixture.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, ForNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, IfThenElseNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { collectAllBlockNames } from '../../../src/compiler/schedule/block_analysis.js';
import { Schedule } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { classifyBlock } from '../../../src/compiler/schedule/rules.js';
import { createSSRSRSTilingSketch } from '../../../src/compiler/autotune/tiling.js';
import { CPU_TILING_SSRSRS } from '../../../src/compiler/autotune/tile_structure.js';
import { getSketchesForBlock } from '../../../src/compiler/autotune/search_space.js';
import { Autotuner } from '../../../src/compiler/autotune/autotuner.js';

function matmulPrimFunc(name, M, N, K) {
  const A = new Buffer('A', [M, K], 'f32', 'global');
  const B = new Buffer('B', [K, N], 'f32', 'global');
  const C = new Buffer('C', [M, N], 'f32', 'global');
  const m = new VariableNode('m', 'int32');
  const n = new VariableNode('n', 'int32');
  const k = new VariableNode('k', 'int32');
  const vm = new VariableNode('vm', 'int32');
  const vn = new VariableNode('vn', 'int32');
  const vk = new VariableNode('vk', 'int32');
  const prod = new MathOpNode('*', new BufferLoadNode(A, [vm, vk]), new BufferLoadNode(B, [vk, vn]));
  const acc = new MathOpNode('+', new BufferLoadNode(C, [vm, vn]), prod);
  const block = new BlockNode(name,
    [spatialIter(vm, m), spatialIter(vn, n), reduceIter(vk, k)],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }],
    new BufferStoreNode(C, [vm, vn], acc), new BufferStoreNode(C, [vm, vn], new IntImmNode(0)));
  let nest = block;
  for (const [v, e] of [[k, K], [n, N], [m, M]]) {
    nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  }
  return new PrimFunc(name, [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

function reference(a, b, M, N, K) {
  const c = new Float32Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n];
    c[m * N + n] = s;
  }
  return c;
}

function compileRun(func, a, b, M, N) {
  const src = new BackendPipeline(CPUTarget()).compile(func).source;
  const fn = new Function('return ' + src)();
  const c = new Float32Array(M * N);
  fn(a, b, c);
  return c;
}

describe('rfactor schedule primitive', () => {
  const CASES = [[3, 4, 8], [5, 2, 12], [4, 4, 16], [1, 6, 9]];
  for (const [M, N, K] of CASES) {
    for (const factor of [2, 3, 4].filter(f => K % f === 0 && f < K)) {
      it(`decomposes ${M}x${N}x${K} reduction (factor ${factor}) into a parallel-reduction + combine that matches baseline`, () => {
        const sch = new Schedule(matmulPrimFunc('g', M, N, K));
        const partial = sch.rfactor('g', 'k', factor);

        expect(partial.shape).toEqual([factor, M, N]);
        expect(ScheduleValidator.validate(sch.func)).toHaveLength(0);

        const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 1.3));
        const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.7));
        const out = compileRun(sch.func, a, b, M, N);
        const ref = reference(a, b, M, N, K);
        for (let i = 0; i < M * N; i++) {
          expect(Math.abs(out[i] - ref[i]) / (1 + Math.abs(ref[i]))).toBeLessThan(1e-4);
        }
      });
    }
  }

  it('rejects a factor that does not divide the reduction extent', () => {
    const sch = new Schedule(matmulPrimFunc('g', 4, 4, 8));
    expect(() => sch.rfactor('g', 'k', 3)).toThrow();
    expect(() => sch.rfactor('g', 'k', 8)).toThrow();
  });

  it('rejects a non-associative accumulation op (subtract), which a tree-split would miscompute', () => {
    const M = 2, N = 2, K = 4;
    const A = new Buffer('A', [M, K], 'f32', 'global');
    const B = new Buffer('B', [K, N], 'f32', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const m = new VariableNode('m', 'int32');
    const n = new VariableNode('n', 'int32');
    const k = new VariableNode('k', 'int32');
    const prod = new MathOpNode('*', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n]));
    const acc = new MathOpNode('-', new BufferLoadNode(C, [m, n]), prod);
    const block = new BlockNode('g',
      [spatialIter(m), spatialIter(n), reduceIter(k)],
      [{ buffer: A }, { buffer: B }], [{ buffer: C }],
      new BufferStoreNode(C, [m, n], acc), new BufferStoreNode(C, [m, n], new IntImmNode(0)));
    let nest = block;
    for (const [v, e] of [[k, K], [n, N], [m, M]]) {
      nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
    }
    const sch = new Schedule(new PrimFunc('g', [], nest, new Map([['A', A], ['B', B], ['C', C]])));
    expect(() => sch.rfactor('g', 'k', 2)).toThrow(/associative/);
  });
});

describe('decomposeReduction + canonical SSRSRS tiling', () => {
  it('decomposeReduction hoists the init so the reduction can be reordered freely, staying correct', () => {
    const M = 4, N = 5, K = 6;
    const sch = new Schedule(matmulPrimFunc('g', M, N, K));
    sch.decomposeReduction('g');
    expect(ScheduleValidator.validate(sch.func)).toHaveLength(0);
    const upd = sch.getLoops('g_upd');
    sch.reorder(upd[0], upd[2], upd[1]);
    expect(ScheduleValidator.validate(sch.func)).toHaveLength(0);
    const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 1.3));
    const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.7));
    const out = compileRun(sch.func, a, b, M, N);
    const ref = reference(a, b, M, N, K);
    for (let i = 0; i < M * N; i++) {
      expect(Math.abs(out[i] - ref[i]) / (1 + Math.abs(ref[i]))).toBeLessThan(1e-4);
    }
  });

  it('the SSRSRS sketch (decompose + interleaved multi-level tiling) is bit-correct across its parameter space', () => {
    const M = 8, N = 8, K = 8;
    const info = classifyBlock(matmulPrimFunc('g', M, N, K), 'g');
    const sketch = createSSRSRSTilingSketch(info, CPU_TILING_SSRSRS);
    expect(sketch).not.toBeNull();
    const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 1.3));
    const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.7));
    const ref = reference(a, b, M, N, K);
    const pick = (i) => {
      const p = {};
      for (const v of sketch.variables) p[v.name] = v.candidates[i % v.candidates.length];
      return p;
    };
    for (let i = 0; i < 12; i++) {
      const params = pick(i);
      const sch = new Schedule(matmulPrimFunc('g', M, N, K));
      sketch.instantiate(params)(sch, 'g', CPUTarget());
      expect(ScheduleValidator.validate(sch.func), JSON.stringify(params)).toHaveLength(0);
      const out = compileRun(sch.func, a, b, M, N);
      for (let j = 0; j < M * N; j++) {
        expect(Math.abs(out[j] - ref[j]) / (1 + Math.abs(ref[j])), JSON.stringify(params)).toBeLessThan(1e-4);
      }
    }
  });

  it('SSRSRS output is correct even when the output buffer is NOT pre-zeroed (the hoisted init must survive codegen)', () => {
    const M = 8, N = 8, K = 8;
    const info = classifyBlock(matmulPrimFunc('g', M, N, K), 'g');
    const sketch = createSSRSRSTilingSketch(info, CPU_TILING_SSRSRS);
    const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 1.3));
    const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.7));
    const ref = reference(a, b, M, N, K);
    const params = {};
    for (const v of sketch.variables) params[v.name] = v.candidates[1];
    const sch = new Schedule(matmulPrimFunc('g', M, N, K));
    sketch.instantiate(params)(sch, 'g', CPUTarget());
    const fn = new Function('return ' + new BackendPipeline(CPUTarget()).compile(sch.func).source)();
    const out = new Float32Array(M * N).fill(99);
    fn(a, b, out);
    for (let j = 0; j < M * N; j++) {
      expect(Math.abs(out[j] - ref[j]) / (1 + Math.abs(ref[j]))).toBeLessThan(1e-4);
    }
  });
});

describe('fuseConsumer (tiling-with-fusion)', () => {
  const M = 3, N = 4, K = 5;
  function twoBlockMatmulThenScale() {
    const A = new Buffer('A', [M, K], 'f32', 'global');
    const B = new Buffer('B', [K, N], 'f32', 'global');
    const T = new Buffer('T', [M, N], 'f32', 'global');
    const O = new Buffer('O', [M, N], 'f32', 'global');
    const m = new VariableNode('m', 'int32'), n = new VariableNode('n', 'int32'), k = new VariableNode('k', 'int32');
    const pAcc = new MathOpNode('+', new BufferLoadNode(T, [m, n]),
      new MathOpNode('*', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n])));
    const P = new BlockNode('P', [spatialIter(m), spatialIter(n), reduceIter(k)],
      [{ buffer: A }, { buffer: B }], [{ buffer: T }], new BufferStoreNode(T, [m, n], pAcc), new BufferStoreNode(T, [m, n], new IntImmNode(0)));
    let pNest = P;
    for (const [v, e] of [[k, K], [n, N], [m, M]]) pNest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, pNest);
    const mc = new VariableNode('mc', 'int32'), nc = new VariableNode('nc', 'int32');
    const C = new BlockNode('C', [spatialIter(mc), spatialIter(nc)],
      [{ buffer: T }], [{ buffer: O }], new BufferStoreNode(O, [mc, nc], new MathOpNode('*', new BufferLoadNode(T, [mc, nc]), new IntImmNode(2))));
    let cNest = C;
    for (const [v, e] of [[nc, N], [mc, M]]) cNest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, cNest);
    return new PrimFunc('f', [], new SeqNode([pNest, cNest]), new Map([['A', A], ['B', B], ['O', O]]));
  }

  it('fuses an elementwise consumer into the matmul producer spatial loops, staying bit-correct (even with garbage output)', () => {
    const sch = new Schedule(twoBlockMatmulThenScale());
    sch.fuseConsumer('P', 'C');
    expect(ScheduleValidator.validate(sch.func)).toHaveLength(0);
    const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 1.3));
    const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.7));
    const out = new Float32Array(M * N).fill(123);
    const fn = new Function('return ' + new BackendPipeline(CPUTarget()).compile(sch.func).source)();
    fn(a, b, out);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n];
      expect(Math.abs(out[m * N + n] - s * 2) / (1 + Math.abs(s * 2))).toBeLessThan(1e-4);
    }
  });

  it('the DAG derivation offers a fused sketch for a producer with a single canonical elementwise consumer', () => {
    const names = getSketchesForBlock(twoBlockMatmulThenScale(), 'P', CPUTarget(), null, {}).map(s => s.name);
    expect(names).toContain('fused');
  });

  it('does NOT offer fusion for a standalone matmul (no consumer)', () => {
    const standalone = (() => {
      const f = twoBlockMatmulThenScale();
      f.body.stmts = [f.body.stmts[0]];
      return f;
    })();
    const names = getSketchesForBlock(standalone, 'P', CPUTarget(), null, {}).map(s => s.name);
    expect(names).not.toContain('fused');
  });

  it('end-to-end: the autotuner selects+applies the fused schedule and the result is bit-correct', () => {
    const tgt = CPUTarget();
    const at = new Autotuner(tgt, {
      strategy: 'evolutionary', populationSize: 6, numGenerations: 2, seed: 3,
      timeBudgetMs: 60000, topKForBenchmark: 4, maxRoundsPerTask: 3, useTuningDB: false,
    });
    at.benchmarkRunner = {
      run(pf) {
        const loops = (new BackendPipeline(tgt).compile(pf).source.match(/for /g) || []).length;
        return { medianMs: loops, minMs: loops };
      },
    };
    const res = at.tuneAndApply(twoBlockMatmulThenScale());
    expect(res.applied).toBe(true);
    expect(res.results.get('P').sketchName).toBe('fused');
    expect(collectAllBlockNames(res.func.body)).toContain('C_fused');

    const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 1.3));
    const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.7));
    const out = new Float32Array(M * N).fill(-999);
    const fn = new Function('return ' + new BackendPipeline(tgt).compile(res.func).source)();
    fn(a, b, out);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n];
      expect(Math.abs(out[m * N + n] - s * 2) / (1 + Math.abs(s * 2))).toBeLessThan(1e-4);
    }
  });

  it('refuses fusion when the consumer WRITES a buffer the producer READS (would reorder a WAR dependence)', () => {
    const A = new Buffer('A', [M, K], 'f32', 'global');
    const B = new Buffer('B', [K, N], 'f32', 'global');
    const T = new Buffer('T', [M, N], 'f32', 'global');
    const m = new VariableNode('m', 'int32'), n = new VariableNode('n', 'int32'), k = new VariableNode('k', 'int32');
    const pAcc = new MathOpNode('+', new BufferLoadNode(T, [m, n]),
      new MathOpNode('*', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n])));
    const P = new BlockNode('P', [spatialIter(m), spatialIter(n), reduceIter(k)],
      [{ buffer: A }, { buffer: B }], [{ buffer: T }], new BufferStoreNode(T, [m, n], pAcc), new BufferStoreNode(T, [m, n], new IntImmNode(0)));
    let pNest = P;
    for (const [v, e] of [[k, K], [n, N], [m, M]]) pNest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, pNest);
    const mc = new VariableNode('mc', 'int32'), nc = new VariableNode('nc', 'int32');
    const C = new BlockNode('C', [spatialIter(mc), spatialIter(nc)],
      [{ buffer: T }], [{ buffer: B }], new BufferStoreNode(B, [mc, nc], new MathOpNode('*', new BufferLoadNode(T, [mc, nc]), new IntImmNode(2))));
    let cNest = C;
    for (const [v, e] of [[nc, N], [mc, M]]) cNest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, cNest);
    const func = new PrimFunc('f', [], new SeqNode([pNest, cNest]), new Map([['A', A], ['B', B]]));
    const names = getSketchesForBlock(func, 'P', CPUTarget(), null, {}).map(s => s.name);
    expect(names).not.toContain('fused');
  });

  it('cacheRead stages a read buffer into a cache and stays bit-correct (and emits the staging copy)', () => {
    const m2 = 4, n2 = 3, k2 = 5;
    const buildMM = () => {
      const A = new Buffer('A', [m2, k2], 'f32', 'global');
      const Bb = new Buffer('B', [k2, n2], 'f32', 'global');
      const C = new Buffer('C', [m2, n2], 'f32', 'global');
      const mi = new VariableNode('m', 'int32'), ni = new VariableNode('n', 'int32'), ki = new VariableNode('k', 'int32');
      const acc = new MathOpNode('+', new BufferLoadNode(C, [mi, ni]),
        new MathOpNode('*', new BufferLoadNode(A, [mi, ki]), new BufferLoadNode(Bb, [ki, ni])));
      const blk = new BlockNode('mm', [spatialIter(mi), spatialIter(ni), reduceIter(ki)],
        [{ buffer: A }, { buffer: Bb }], [{ buffer: C }], new BufferStoreNode(C, [mi, ni], acc), new BufferStoreNode(C, [mi, ni], new IntImmNode(0)));
      let nest = blk;
      for (const [v, e] of [[ki, k2], [ni, n2], [mi, m2]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
      return new PrimFunc('f', [], nest, new Map([['A', A], ['B', Bb], ['C', C]]));
    };
    const a = Float32Array.from({ length: m2 * k2 }, (_, i) => Math.sin(i * 1.1));
    const b = Float32Array.from({ length: k2 * n2 }, (_, i) => Math.cos(i * 0.6));
    const expectMatmul = (src2) => {
      const out = new Float32Array(m2 * n2).fill(9);
      new Function('return ' + src2)()(a, b, out);
      for (let mm = 0; mm < m2; mm++) for (let nn = 0; nn < n2; nn++) {
        let s = 0; for (let kk = 0; kk < k2; kk++) s += a[mm * k2 + kk] * b[kk * n2 + nn];
        expect(Math.abs(out[mm * n2 + nn] - s)).toBeLessThan(1e-5);
      }
    };
    const sch = new Schedule(buildMM());
    sch.cacheRead('mm', 'A', 'local');
    expect(ScheduleValidator.validate(sch.func)).toHaveLength(0);
    const src = new BackendPipeline(CPUTarget()).compile(sch.func).source;
    expect(src).toMatch(/_cache/);
    expectMatmul(src);

    const schW = new Schedule(buildMM());
    schW.cacheWrite('mm', 'C', 'local');
    expect(ScheduleValidator.validate(schW.func)).toHaveLength(0);
    expectMatmul(new BackendPipeline(CPUTarget()).compile(schW.func).source);
  });

  it('fuseConsumer fails loudly (no silent duplicate) when the consumer nest is not a direct SeqNode sibling', () => {
    const f = twoBlockMatmulThenScale();
    const cNest = f.body.stmts[1];
    const guarded = new IfThenElseNode(new MathOpNode('<', new IntImmNode(0), new IntImmNode(1)), cNest, null);
    f.body.stmts = [f.body.stmts[0], guarded];
    const sch = new Schedule(f);
    expect(() => sch.fuseConsumer('P', 'C')).toThrow();
  });
});
