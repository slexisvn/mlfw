import { describe, it, expect } from 'vitest';
import {
  tensor, Linear, Sequential, ReLU, add,
} from '../../src/index.js';
import { compile } from '../../src/tracing/compile.js';
import { WasmTarget } from '../../src/backend/target.js';
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
