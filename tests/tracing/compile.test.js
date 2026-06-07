import { describe, it, expect } from 'vitest';
import {
  tensor, Linear, Sequential, ReLU,
} from '../../src/index.js';
import { compile } from '../../src/tracing/compile.js';
import { Tensor } from '../../src/tensor/core/tensor.js';

describe('compile returns executable tensors', () => {
  it('returns a Tensor with correct shape and dtype', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x]);

    const out = compiled(x);
    expect(out).toBeInstanceOf(Tensor);
    expect(out.shape).toEqual([1, 2]);
    expect(out.dtype).toBe('f32');
  });

  it('numerical output matches eager forward pass', () => {
    const model = new Sequential(new Linear(3, 2));
    const x = tensor([[1, 2, 3]]);

    const eager = model.forward(x);
    const compiled = compile(model, [x]);
    const compiled_out = compiled(x);

    expect(compiled_out.shape).toEqual(eager.shape);
    const eagerData = eager.data;
    const compiledData = compiled_out.data;
    for (let i = 0; i < eagerData.length; i++) {
      expect(compiledData[i]).toBeCloseTo(eagerData[i], 4);
    }
  });

  it('multi-layer model produces correct shape', () => {
    const model = new Sequential(
      new Linear(4, 3),
      new ReLU(),
      new Linear(3, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);

    const compiled = compile(model, [x]);
    const out = compiled(x);

    expect(out).toBeInstanceOf(Tensor);
    expect(out.shape).toEqual([1, 2]);
    expect(out.dtype).toBe('f32');
    expect(out.data.length).toBe(2);
    for (let i = 0; i < out.data.length; i++) {
      expect(Number.isFinite(out.data[i])).toBe(true);
    }
  });
});

describe('compile shape-based recompilation cache', () => {
  it('same shape input reuses cached compilation', () => {
    const model = new Sequential(new Linear(4, 2));
    const x1 = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x1]);

    const r1 = compiled(x1);
    const x2 = tensor([[5, 6, 7, 8]]);
    const r2 = compiled(x2);
    expect(r1).toBeInstanceOf(Tensor);
    expect(r2).toBeInstanceOf(Tensor);
    expect(r1.shape).toEqual(r2.shape);
  });

  it('different shape triggers recompilation', () => {
    const model = new Sequential(new Linear(4, 2));
    const x1 = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x1]);
    const r1 = compiled(x1);

    const x2 = tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    const r2 = compiled(x2);
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
  it('compiles on first call when no example inputs given', () => {
    const model = new Sequential(new Linear(4, 2));
    const compiled = compile(model);
    expect(compiled.kernels()).toEqual([]);

    const x = tensor([[1, 2, 3, 4]]);
    const r = compiled(x);
    expect(r).toBeInstanceOf(Tensor);
    expect(r.shape).toEqual([1, 2]);
    expect(compiled.kernels().length).toBeGreaterThan(0);
  });
});
