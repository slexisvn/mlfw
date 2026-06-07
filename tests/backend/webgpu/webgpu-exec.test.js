import { describe, it, expect } from 'vitest';
import {
  tensor, Linear, Sequential, ReLU, Sigmoid, Tanh,
  compile, WebGPUTarget,
} from '../../../src/index.js';

function compileWebGPU(model, inputs) {
  return compile(model, inputs, { target: WebGPUTarget() });
}

describe('webgpu compilation output', () => {
  it('source contains no CUDA artifacts', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).not.toMatch(/__global__/);
    expect(src).not.toMatch(/threadIdx/);
    expect(src).not.toMatch(/blockIdx/);
    expect(src).not.toMatch(/#include/);
    expect(src).not.toMatch(/float\*/);
  });

  it('source has balanced braces', () => {
    const model = new Sequential(new Linear(4, 2), new ReLU());
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();
    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('source has @compute and @workgroup_size', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).toMatch(/@compute/);
    expect(src).toMatch(/@workgroup_size\(\d+/);
  });

  it('bindings use correct access modes', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).toMatch(/var<storage, read>/);
    expect(src).toMatch(/var<storage, read_write>/);
  });

  it('kernels() returns non-empty list', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    expect(compiled.kernels().length).toBeGreaterThan(0);
  });

  it('source returns valid WGSL string', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();
    expect(typeof src).toBe('string');
    expect(src).toContain('fn ');
    expect(src).toContain('array<f32>');
  });

  it('deep model emits all local var declarations', () => {
    const model = new Sequential(
      new Linear(8, 16), new ReLU(),
      new Linear(16, 8), new ReLU(),
      new Linear(8, 4),
    );
    const x = tensor([[1, 2, 3, 4, 5, 6, 7, 8]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    const usedVars = new Set();
    const declaredVars = new Set();
    for (const m of src.matchAll(/\b(buf_\d+)\[/g)) usedVars.add(m[1]);
    for (const m of src.matchAll(/var\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);
    for (const m of src.matchAll(/var<storage[^>]*>\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);

    for (const v of usedVars) {
      expect(declaredVars.has(v)).toBe(true);
    }
  });

  it('activation models use WGSL builtins', () => {
    const model = new Sequential(new Linear(3, 3), new Sigmoid());
    const x = tensor([[1, 2, 3]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();
    expect(src).toMatch(/exp\(/);
  });

  it('multi-layer with mixed activations compiles', () => {
    const model = new Sequential(
      new Linear(4, 8), new ReLU(),
      new Linear(8, 8), new Sigmoid(),
      new Linear(8, 4), new Tanh(),
      new Linear(4, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).toContain('@compute');
    expect(src).toContain('max(');
    expect(src).toContain('exp(');
    expect(src).toContain('tanh(');

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });
});
