import { describe, it, expect } from 'vitest';
import {
  tensor, Linear, Sequential, ReLU, add, relu, sum, LSTM, GRU,
} from '../../src/index.js';
import { compile, _traceCore } from '../../src/tracing/compile.js';
import { foldWeightParams } from '../../src/tracing/fold_params.js';
import { tensorToContiguous } from '../../src/dispatcher/jit_dispatch.js';
import { WasmTarget, CPUTarget } from '../../src/backend/target.js';
import { QuantizationScheme } from '../../src/compiler/ir/graph/quantization_types.js';
import { Tensor } from '../../src/tensor/core/tensor.js';

describe('compile returns executable tensors', () => {
  it('returns a Tensor with correct shape and dtype', async () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x]);

    const out = await compiled(x);
    expect(out).toBeInstanceOf(Tensor);
    expect(out.shape).toEqual([1, 2]);
    expect(out.dtype).toBe('f32');
  });

  it('numerical output matches eager forward pass', async () => {
    const model = new Sequential(new Linear(3, 2));
    const x = tensor([[1, 2, 3]]);

    const eager = model.forward(x);
    const compiled = compile(model, [x]);
    const compiled_out = await compiled(x);

    expect(compiled_out.shape).toEqual(eager.shape);
    const eagerData = eager.data;
    const compiledData = compiled_out.data;
    for (let i = 0; i < eagerData.length; i++) {
      expect(compiledData[i]).toBeCloseTo(eagerData[i], 4);
    }
  });

  it('multi-layer model produces correct shape', async () => {
    const model = new Sequential(
      new Linear(4, 3),
      new ReLU(),
      new Linear(3, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);

    const compiled = compile(model, [x]);
    const out = await compiled(x);

    expect(out).toBeInstanceOf(Tensor);
    expect(out.shape).toEqual([1, 2]);
    expect(out.dtype).toBe('f32');
    expect(out.data.length).toBe(2);
    for (let i = 0; i < out.data.length; i++) {
      expect(Number.isFinite(out.data[i])).toBe(true);
    }
  });

  it('compiled keepdim reduction preserves the reduced axis and matches eager', async () => {
    const x = tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    for (const reduce of [(t) => t.sum(1, true), (t) => t.mean(1, true), (t) => t.max(1, true)]) {
      const eager = reduce(x);
      const compiled = compile({ forward: (t) => reduce(t) }, [x]);
      const out = await compiled(x);
      expect(out.shape).toEqual(eager.shape);
      expect(out.shape).toEqual([2, 1]);
      const e = eager.contiguous().data;
      const c = out.contiguous().data;
      for (let i = 0; i < e.length; i++) expect(c[i]).toBeCloseTo(e[i], 4);
    }
  });
});

describe('compile shape-based recompilation cache', () => {
  it('same shape input reuses cached compilation', async () => {
    const model = new Sequential(new Linear(4, 2));
    const x1 = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x1]);

    const r1 = await compiled(x1);
    const x2 = tensor([[5, 6, 7, 8]]);
    const r2 = await compiled(x2);
    expect(r1).toBeInstanceOf(Tensor);
    expect(r2).toBeInstanceOf(Tensor);
    expect(r1.shape).toEqual(r2.shape);
  });

  it('different shape triggers recompilation', async () => {
    const model = new Sequential(new Linear(4, 2));
    const x1 = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x1]);
    const r1 = await compiled(x1);

    const x2 = tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    const r2 = await compiled(x2);
    expect(r1.shape).toEqual([1, 2]);
    expect(r2.shape).toEqual([2, 2]);
  });
});

describe('compile accessor methods', () => {
  it('.graph() returns a GraphModule with functions', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x]);
    const graph = compiled.graph();
    const func = graph.functions().next().value;
    expect(func).toBeDefined();
    expect(func.inputTypes.length).toBeGreaterThan(0);
  });

  it('.kernels() returns non-empty kernel list', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x]);
    expect(compiled.kernels().length).toBeGreaterThan(0);
  });

  it('.source() returns generated code string', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x]);
    const src = compiled.source();
    expect(typeof src).toBe('string');
    expect(src.length).toBeGreaterThan(0);
  });

  it('.original holds reference to the model', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x]);
    expect(compiled.original).toBe(model);
  });
});

describe('compile lazy (no exampleInputs)', () => {
  it('compiles on first call when no example inputs given', async () => {
    const model = new Sequential(new Linear(4, 2));
    const compiled = compile(model);
    expect(compiled.kernels()).toEqual([]);

    const x = tensor([[1, 2, 3, 4]]);
    const r = await compiled(x);
    expect(r).toBeInstanceOf(Tensor);
    expect(r.shape).toEqual([1, 2]);
    expect(compiled.kernels().length).toBeGreaterThan(0);
  });
});

describe('compile failures carry a reproduction context', () => {
  it('attaches input shapes/dtypes, target, and config on a compile error', () => {
    const model = { forward: (a, b) => add(a, b) };
    const a = tensor([[1, 2, 3]]);
    const b = tensor([[1, 2, 3, 4]]);

    let err = null;
    try {
      compile(model, [a, b], { target: WasmTarget() });
    } catch (e) { err = e; }

    expect(err).not.toBeNull();
    expect(err.repro).toBeDefined();
    expect(err.repro.phase).toBe('compile');
    expect(err.repro.target).toBe('wasm_generic');
    expect(err.repro.inputs).toEqual([
      { shape: [1, 3], dtype: 'f32' },
      { shape: [1, 4], dtype: 'f32' },
    ]);
  });
});

describe('shapeBuckets: static specialization with dynamic fallback', () => {
  const mk = (rows, cols) => tensor(Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => ((i * cols + j) % 13) / 13 - 0.4)));
  const fwd = (x) => sum(relu(x), 1);

  it('bucket shapes use a static kernel, off-bucket shapes use the dynamic fallback, all correct', async () => {
    const compiled = compile({ forward: (a) => fwd(a) }, [mk(3, 4)], {
      target: CPUTarget(),
      dynamic_shapes: [new Set([0])],
      shapeBuckets: [[[8, 4]], [[16, 4]]],
    });

    expect(compiled.source()).not.toMatch(/_ds/);

    for (const N of [3, 8, 16, 7]) {
      const x = mk(N, 4);
      const eager = Array.from(fwd(x).contiguous().data);
      const out = Array.from((await compiled(x)).data);
      expect(out.length).toBe(eager.length);
      for (let i = 0; i < eager.length; i++) {
        expect(Math.abs(eager[i] - out[i]), `N=${N} idx ${i}`).toBeLessThan(1e-4);
      }
    }
  });
});

describe('foldWeightParams folds captured weight params into constants', () => {
  it('folds rank-2 float weight params, prunes them from capturedParams and entry args', () => {
    const lin = new Linear(4, 3, false);
    const model = new Sequential(lin);
    const x = tensor([[0.5, -0.3, 0.8, 0.1]]);

    const traced = _traceCore((...a) => model.forward(...a), [x], { name: 'fold' });
    expect(traced.capturedParams.length).toBe(1);
    const func = traced.graph.functions().next().value;
    const argsBefore = func.entryBlock.arguments.length;

    const folded = foldWeightParams(traced, tensorToContiguous);
    expect(folded.capturedParams.length).toBe(0);
    const func2 = folded.graph.functions().next().value;
    expect(func2.entryBlock.arguments.length).toBe(argsBefore - 1);
    expect(func2.inputTypes.length).toBe(func2.entryBlock.arguments.length);
  });

  it('compiling with foldWeights produces identical output to the baseline', async () => {
    const model = new Sequential(new Linear(4, 5, false), new ReLU(), new Linear(5, 3, false));
    const x = tensor([[0.5, -0.3, 0.8, 0.1], [0.2, 0.9, -0.4, 0.6]]);

    const baseline = compile(model, [x]);
    const folded = compile(model, [x], { foldWeights: true });
    const ob = await baseline(x);
    const of = await folded(x);

    expect(of.shape).toEqual(ob.shape);
    for (let i = 0; i < ob.data.length; i++) {
      expect(of.data[i]).toBeCloseTo(ob.data[i], 5);
    }
    expect(folded.source()).not.toBe(baseline.source());
  });
});

describe('per-channel int8 quantization preserves top-1 on a real traced classifier', () => {
  it('quantized top-1 matches float top-1 and the ground-truth labels', async () => {
    const D = 6, C = 4, NTEST = 40;
    const proto = [];
    for (let c = 0; c < C; c++) {
      const p = [];
      for (let d = 0; d < D; d++) p.push((d === c % D ? 1 : 0.15) * Math.cos(c * 1.7 + d * 0.9) * (c + 1));
      proto.push(p);
    }
    const lin = new Linear(D, C, false);
    const model = new Sequential(lin);
    for (let c = 0; c < C; c++) for (let d = 0; d < D; d++) lin.weight.data[c * D + d] = proto[c][d];

    let seed = 12345;
    const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 0x100000000; };
    const xs = [], labels = [];
    for (let i = 0; i < NTEST; i++) {
      const c = i % C;
      const row = [];
      for (let d = 0; d < D; d++) row.push(proto[c][d] + (rnd() - 0.5) * 0.25);
      xs.push(row);
      labels.push(c);
    }
    const X = tensor(xs);

    const floatModel = compile(model, [X]);
    const quantModel = compile(model, [X], {
      foldWeights: true,
      quantization: { enabled: true, scheme: QuantizationScheme.PER_CHANNEL, quantizableOps: new Set(['dot']) },
    });
    const fo = await floatModel(X);
    const qo = await quantModel(X);

    const top1 = (data) => {
      const r = [];
      for (let i = 0; i < NTEST; i++) {
        let bi = 0, bv = -Infinity;
        for (let c = 0; c < C; c++) { const v = data[i * C + c]; if (v > bv) { bv = v; bi = c; } }
        r.push(bi);
      }
      return r;
    };
    const fp = top1(fo.data), qp = top1(qo.data);
    let fAcc = 0, qAcc = 0, agree = 0;
    for (let i = 0; i < NTEST; i++) {
      if (fp[i] === labels[i]) fAcc++;
      if (qp[i] === labels[i]) qAcc++;
      if (fp[i] === qp[i]) agree++;
    }
    expect(fAcc).toBe(NTEST);
    expect(qAcc).toBe(NTEST);
    expect(agree).toBe(NTEST);
  });
});

describe('RNN modules compile without an explicit initial state', () => {
  const mkInput = (B, T, E) => tensor(Array.from({ length: B * T * E }, (_, i) => Math.sin(i * 0.01) * 0.3), { shape: [B, T, E] });
  for (const [name, Mod] of [['LSTM', LSTM], ['GRU', GRU]]) {
    for (const Target of [CPUTarget, WasmTarget]) {
      it(`${name} on ${Target.name} matches eager forward`, async () => {
        const m = new Mod(8, 12, 2, true);
        const x = mkInput(2, 5, 8);
        const eager = m.forward(x)[0];
        const cf = compile({ forward: (i) => m.forward(i)[0] }, [x], { target: Target() });
        const out = await cf(x);
        expect(out.shape).toEqual(eager.shape);
        const e = Array.from(eager.contiguous().data), o = Array.from(out.contiguous().data);
        for (let i = 0; i < e.length; i++) expect(o[i]).toBeCloseTo(e[i], 4);
      });
    }
  }
});
