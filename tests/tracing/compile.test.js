import { describe, it, expect } from 'vitest';
import {
  tensor, Linear, Sequential, ReLU,
} from '../../src/index.js';
import { compile } from '../../src/tracing/compile.js';

describe('compile shape-based recompilation cache', () => {
  it('same shape input reuses cached compilation', () => {
    const model = new Sequential(new Linear(4, 2));
    const x1 = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x1]);

    const r1 = compiled(x1);
    const x2 = tensor([[5, 6, 7, 8]]);
    const r2 = compiled(x2);
    expect(r1.result).toBe(r2.result);
  });

  it('different shape triggers recompilation', () => {
    const model = new Sequential(new Linear(4, 2));
    const x1 = tensor([[1, 2, 3, 4]]);
    const compiled = compile(model, [x1]);
    const r1 = compiled(x1);

    const x2 = tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    const r2 = compiled(x2);
    expect(r2.result).not.toBe(r1.result);
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
    expect(r.result).toBeDefined();
    expect(compiled.kernels().length).toBeGreaterThan(0);
  });
});
