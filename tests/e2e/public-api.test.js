import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import * as api from '../../src/index.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const pkg = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf8'));

const REQUIRED = {
  tensor: ['tensor', 'Tensor', 'zeros', 'ones', 'randn', 'randperm', 'arange', 'eye', 'linspace'],
  ops: ['add', 'sub', 'mul', 'div', 'matmul', 'relu', 'gelu', 'softmax', 'log_softmax', 'sum', 'mean', 'max', 'min'],
  nn: ['Linear', 'Sequential', 'ReLU', 'Sigmoid', 'Tanh', 'LayerNorm', 'Softmax', 'Dropout'],
  train: ['Adam', 'MSELoss', 'TensorDataset', 'DataLoader'],
  compile: ['compile', 'trace', 'compileWithBackward', 'CPUTarget', 'WasmTarget', 'CUDATarget', 'WebGPUTarget'],
  runtime: ['preloadWebGPU', 'preloadCudaRuntime', 'dispatcher'],
  random: ['manual_seed', 'seed', 'unseed'],
  tokenizer: ['Tokenizer', 'Vocab'],
};

describe('public API surface (package entry point)', () => {
  for (const [group, names] of Object.entries(REQUIRED)) {
    it(`exports the ${group} surface`, () => {
      const missing = names.filter((n) => api[n] === undefined);
      expect(missing, `src/index.ts stopped exporting: ${missing.join(', ')}`).toEqual([]);
    });
  }

  it('exports no undefined bindings', () => {
    const dead = Object.keys(api).filter((k) => api[k] === undefined);
    expect(dead, `barrel exports resolve to undefined: ${dead.join(', ')}`).toEqual([]);
  });

  it('a full compile+run works using only barrel imports', async () => {
    api.manual_seed(3);
    const model = new api.Sequential(new api.Linear(4, 8), new api.ReLU(), new api.Linear(8, 2));
    const x = api.tensor([[1, 2, 3, 4]]);

    const eager = model.forward(x);
    const compiled = api.compile(model, [x], { target: api.CPUTarget() });
    const out = await compiled(x);

    expect(out.shape).toEqual([1, 2]);
    expect(compiled.kernels().length).toBeGreaterThan(0);
    for (let i = 0; i < out.data.length; i++) expect(out.data[i]).toBeCloseTo(eager.data[i], 4);
  });
});

describe('package manifest is publishable', () => {
  it('every published entry point lives under a path in "files"', () => {
    const roots = pkg.files.map((f) => f.replace(/\/$/, ''));
    const entries = [pkg.main, pkg.types, ...Object.values(pkg.exports['.'])].filter((e) => typeof e === 'string' && e.startsWith('./'));
    for (const e of entries) {
      const rel = e.slice(2);
      expect(roots.some((r) => rel === r || rel.startsWith(r + '/')), `${e} is not covered by "files": ${roots.join(', ')}`).toBe(true);
    }
  });

  it('every #io subpath target exists in src', () => {
    for (const [spec, targets] of Object.entries(pkg.imports)) {
      for (const [condition, target] of Object.entries(targets)) {
        const probe = target.replace('*', spec.endsWith('/*') ? 'device.js' : '');
        const path = resolve(ROOT, probe);
        const ok = existsSync(path) || existsSync(path.replace(/\.js$/, '.ts')) || existsSync(dirname(path));
        expect(ok, `${spec} (${condition}) points at a missing file: ${target}`).toBe(true);
      }
    }
  });

  it('declares the test scripts the projects define', () => {
    for (const name of ['test', 'test:unit', 'test:e2e', 'test:cuda', 'test:webgpu', 'test:stress', 'test:perf']) {
      expect(pkg.scripts[name], `package.json is missing the ${name} script`).toBeTruthy();
    }
  });
});
