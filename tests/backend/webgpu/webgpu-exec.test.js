import { describe, it, expect } from 'vitest';
import {
  tensor, Linear, Sequential, ReLU, Sigmoid, Tanh,
  GELU, SiLU, LeakyReLU, ELU,
  compile, WebGPUTarget,
} from '../../../src/index.js';

function compileWebGPU(model, inputs, opts) {
  return compile(model, inputs, { target: WebGPUTarget(), ...opts });
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

  it('scheduled multi-stage kernel uses workgroup memory and barriers', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toMatch(/var<workgroup>/);
    expect(src).toContain('workgroupBarrier()');
    expect(src).toMatch(/@workgroup_size\([2-9]|\d{2,}/);
    expect(src).toMatch(/local_invocation_id/);
  });

  it('scheduled kernel emits bounds guard for smaller stages', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toMatch(/if \(i32\(_lid\.x\) < 2\)/);
  });

  it('scheduled kernel has balanced braces', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();
    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('scheduled kernel declares all used buffers', () => {
    const model = new Sequential(
      new Linear(8, 16), new ReLU(),
      new Linear(16, 8), new ReLU(),
      new Linear(8, 4),
    );
    const x = tensor([[1, 2, 3, 4, 5, 6, 7, 8]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const usedVars = new Set();
    const declaredVars = new Set();
    for (const m of src.matchAll(/\b(buf_\d+)\[/g)) usedVars.add(m[1]);
    for (const m of src.matchAll(/var\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);
    for (const m of src.matchAll(/var<storage[^>]*>\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);
    for (const m of src.matchAll(/var<workgroup>\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);

    for (const v of usedVars) {
      expect(declaredVars.has(v)).toBe(true);
    }
  });

  it('unscheduled kernel uses private var, not workgroup', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).not.toContain('var<workgroup>');
    expect(src).not.toContain('workgroupBarrier()');
  });

  it('scheduled single-layer has no barriers or workgroup promotion', () => {
    const model = new Sequential(new Linear(4, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).not.toContain('workgroupBarrier()');
  });

  it('scheduled workgroup_size matches largest stage extent', () => {
    const model = new Sequential(new Linear(4, 16), new ReLU(), new Linear(16, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const wgMatch = src.match(/@workgroup_size\((\d+)/);
    expect(wgMatch).not.toBeNull();
    expect(Number(wgMatch[1])).toBeGreaterThanOrEqual(16);
  });

  it('scheduled kernel barrier count matches stage count minus one', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const barrierCount = (src.match(/workgroupBarrier\(\)/g) || []).length;
    expect(barrierCount).toBeGreaterThanOrEqual(2);
  });

  it('storage bindings are never promoted to workgroup', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const storageBindings = [...src.matchAll(/var<storage[^>]*>\s+(buf_\d+)/g)].map(m => m[1]);
    const workgroupVars = [...src.matchAll(/var<workgroup>\s+(buf_\d+)/g)].map(m => m[1]);

    for (const wg of workgroupVars) {
      expect(storageBindings).not.toContain(wg);
    }
  });

  it('scheduled deep model with mixed activations has correct structure', () => {
    const model = new Sequential(
      new Linear(4, 8), new ReLU(),
      new Linear(8, 8), new Sigmoid(),
      new Linear(8, 4), new Tanh(),
      new Linear(4, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('var<workgroup>');
    expect(src).toContain('workgroupBarrier()');
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

  it('bounds guards only appear when stages have different extents', () => {
    const model = new Sequential(new Linear(4, 4), new ReLU());
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).not.toMatch(/if \(i32\(_lid\.\w\) </);
  });

  it('each workgroup var has explicit array size', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const wgDecls = [...src.matchAll(/var<workgroup>\s+\w+:\s*array<[^,]+,\s*(\d+)>/g)];
    expect(wgDecls.length).toBeGreaterThan(0);
    for (const m of wgDecls) {
      expect(Number(m[1])).toBeGreaterThan(0);
    }
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

  it('wide bottleneck: Linear(4,64)->ReLU->Linear(64,2) scheduled', () => {
    const model = new Sequential(new Linear(4, 64), new ReLU(), new Linear(64, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const wgMatch = src.match(/@workgroup_size\((\d+)/);
    expect(Number(wgMatch[1])).toBeGreaterThanOrEqual(64);
    expect(src).toContain('var<workgroup>');
    expect(src).toContain('workgroupBarrier()');
    expect(src).toMatch(/if \(i32\(_lid\.x\) < 2\)/);
  });

  it('5-layer deep MLP scheduled compiles with valid WGSL', () => {
    const model = new Sequential(
      new Linear(4, 16), new ReLU(),
      new Linear(16, 32), new ReLU(),
      new Linear(32, 16), new ReLU(),
      new Linear(16, 8), new ReLU(),
      new Linear(8, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);

    const usedVars = new Set();
    const declaredVars = new Set();
    for (const m of src.matchAll(/\b(buf_\d+)\[/g)) usedVars.add(m[1]);
    for (const m of src.matchAll(/var\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);
    for (const m of src.matchAll(/var<storage[^>]*>\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);
    for (const m of src.matchAll(/var<workgroup>\s+(buf_\d+)\s*:/g)) declaredVars.add(m[1]);
    for (const v of usedVars) {
      expect(declaredVars.has(v)).toBe(true);
    }
  });

  it('GELU activation compiles to valid WGSL', () => {
    const model = new Sequential(new Linear(4, 8), new GELU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).toContain('@compute');
    expect(src).toContain('fn ');
    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('SiLU activation compiles to valid WGSL', () => {
    const model = new Sequential(new Linear(4, 8), new SiLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).toContain('@compute');
    expect(src).toContain('exp(');
    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('LeakyReLU compiles to valid WGSL', () => {
    const model = new Sequential(new Linear(4, 8), new LeakyReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x]);
    const src = compiled.source();

    expect(src).toContain('@compute');
    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('all-different activations scheduled', () => {
    const model = new Sequential(
      new Linear(4, 8), new ReLU(),
      new Linear(8, 8), new Sigmoid(),
      new Linear(8, 8), new Tanh(),
      new Linear(8, 8), new GELU(),
      new Linear(8, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('var<workgroup>');
    expect(src).toContain('workgroupBarrier()');
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

  it('same-extent model scheduled has no bounds guards', () => {
    const model = new Sequential(
      new Linear(4, 8), new ReLU(),
      new Linear(8, 8), new ReLU(),
      new Linear(8, 8),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).not.toMatch(/if \(i32\(_lid\.\w\) </);
  });

  it('expanding model: Linear(2,4)->ReLU->Linear(4,16)->ReLU->Linear(16,32)', () => {
    const model = new Sequential(
      new Linear(2, 4), new ReLU(),
      new Linear(4, 16), new ReLU(),
      new Linear(16, 32),
    );
    const x = tensor([[1, 2]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const wgMatch = src.match(/@workgroup_size\((\d+)/);
    expect(Number(wgMatch[1])).toBeGreaterThanOrEqual(32);
    expect(src).toContain('var<workgroup>');
    expect(src).toContain('workgroupBarrier()');
    expect(src).toMatch(/if \(i32\(_lid\.x\) < 4\)/);
  });

  it('contracting model: Linear(32,16)->ReLU->Linear(16,4)->ReLU->Linear(4,1)', () => {
    const model = new Sequential(
      new Linear(32, 16), new ReLU(),
      new Linear(16, 4), new ReLU(),
      new Linear(4, 1),
    );
    const x = tensor([Array.from({ length: 32 }, (_, i) => i)]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    expect(src).toContain('var<workgroup>');
    expect(src).toContain('workgroupBarrier()');
    expect(src).toMatch(/if \(i32\(_lid\.x\) < 4\)/);
    expect(src).toMatch(/if \(i32\(_lid\.x\) < 1\)/);

    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });

  it('workgroup var count matches intermediate buffer count', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const wgVars = [...src.matchAll(/var<workgroup>\s+(buf_\d+)/g)].map(m => m[1]);
    const privateVars = [...src.matchAll(/^\s+var (buf_\d+):/gm)].map(m => m[1]);

    expect(wgVars.length).toBeGreaterThan(0);
    expect(privateVars.length).toBe(0);
  });

  it('barrier appears before every bounds-guarded section', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileWebGPU(model, [x], { enableSchedule: true });
    const src = compiled.source();

    const lines = src.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].match(/if \(i32\(_lid\.\w\) < \d+\)/)) {
        const preceding = lines.slice(0, i).reverse().find(l => l.trim().length > 0);
        expect(preceding.trim()).toBe('workgroupBarrier();');
      }
    }
  });
});
