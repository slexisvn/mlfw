import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execFileSync } from 'child_process';
import { writeFileSync, unlinkSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { tmpdir } from 'os';

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

let _tmpCounter = 0;
let bundlePath = null;

function toFileURL(p) {
  return 'file:///' + p.replace(/\\/g, '/');
}

beforeAll(async () => {
  const esbuild = (await import('esbuild')).default;
  const bundleDir = join(PROJECT_ROOT, 'node_modules', '.tmp');
  mkdirSync(bundleDir, { recursive: true });
  bundlePath = join(bundleDir, `_webgpu_bundle_${Date.now()}.mjs`);
  await esbuild.build({
    stdin: {
      contents: "export * from './src/index.js';\n"
        + "export { CPUTarget, WebGPUTarget } from './src/backend/target.js';\n"
        + "export { conv2d } from './src/nn/functional/conv.js';\n"
        + "export { max_pool2d } from './src/nn/functional/pooling.js';\n"
        + "export { resetDevice } from './src/runtime/webgpu.js';\n",
      resolveDir: PROJECT_ROOT,
      loader: 'js',
    },
    bundle: true, format: 'esm', platform: 'node',
    external: ['webgpu', 'koffi'], outfile: bundlePath,
  });
});

afterAll(() => {
  if (bundlePath) {
    try { unlinkSync(bundlePath); } catch (_) {}
  }
});

const maxErr = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const relErr = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]) / (1 + Math.abs(b[i]))), 0);
const runGPUJSON = (testBody) => JSON.parse(runGPUTest(testBody));

function runGPUTest(testBody) {
  const tmpFile = join(tmpdir(), `_webgpu_test_${Date.now()}_${_tmpCounter++}.mjs`);
  const script = `
import { tensor, Linear, Sequential, ReLU, Sigmoid, Tanh, compile, WebGPUTarget, CPUTarget,
  relu, sum, mean, neg, add, mul, conv2d, max_pool2d, resetDevice } from '${toFileURL(bundlePath)}';
const emit = (v) => process.stdout.write(JSON.stringify(v));
async function main() {
  ${testBody}
  resetDevice();
}
main().then(() => process.exit(0)).catch(e => { process.stderr.write(e.message + '\\n' + e.stack + '\\n'); resetDevice(); process.exit(1); });
`;
  writeFileSync(tmpFile, script, 'utf8');
  try {
    return execFileSync(process.execPath, [tmpFile], {
      cwd: PROJECT_ROOT,
      timeout: 30000,
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe'],
    }).trim();
  } catch (e) {
    const stderr = e.stderr ? e.stderr.toString().trim() : '';
    if (e.signal === 'SIGSEGV' || e.status === 139 || e.status === 3221225477) {
      if (e.stdout && e.stdout.toString().trim()) return e.stdout.toString().trim();
      throw new Error('GPU segfault before output: ' + stderr);
    }
    throw new Error('GPU test failed: ' + stderr);
  } finally {
    try { unlinkSync(tmpFile); } catch (_) {}
  }
}

describe('webgpu GPU execution', () => {
  it('Linear forward [1,4]->[1,2] matches eager', () => {
    const { shape, gpu, eager } = runGPUJSON(`
      const model = new Sequential(new Linear(4, 2));
      const x = tensor([[1, 2, 3, 4]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      emit({ shape: out.shape, gpu: Array.from(out.data), eager: Array.from(eager.data) });
    `);
    expect(shape).toEqual([1, 2]);
    expect(maxErr(gpu, eager)).toBeLessThan(1e-3);
  });

  it('Linear + ReLU clamps negatives', () => {
    const { gpu, eager } = runGPUJSON(`
      const model = new Sequential(new Linear(4, 4), new ReLU());
      const x = tensor([[1, -1, 2, -2]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      emit({ gpu: Array.from(out.data), eager: Array.from(eager.data) });
    `);
    expect(Math.min(...gpu)).toBeGreaterThanOrEqual(0);
    expect(maxErr(gpu, eager)).toBeLessThan(1e-3);
  });

  it('Linear + Sigmoid bounds [0,1]', () => {
    const { gpu, eager } = runGPUJSON(`
      const model = new Sequential(new Linear(3, 3), new Sigmoid());
      const x = tensor([[10, -10, 0]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      emit({ gpu: Array.from(out.data), eager: Array.from(eager.data) });
    `);
    expect(Math.min(...gpu)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...gpu)).toBeLessThanOrEqual(1);
    expect(maxErr(gpu, eager)).toBeLessThan(1e-3);
  });

  it('Linear + Tanh bounds [-1,1]', () => {
    const { gpu, eager } = runGPUJSON(`
      const model = new Sequential(new Linear(3, 3), new Tanh());
      const x = tensor([[5, -5, 0]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      emit({ gpu: Array.from(out.data), eager: Array.from(eager.data) });
    `);
    expect(Math.min(...gpu)).toBeGreaterThanOrEqual(-1);
    expect(Math.max(...gpu)).toBeLessThanOrEqual(1);
    expect(maxErr(gpu, eager)).toBeLessThan(1e-3);
  });

  it('deterministic across repeated runs', () => {
    const { first, second } = runGPUJSON(`
      const model = new Sequential(new Linear(4, 2));
      const x = tensor([[1, 2, 3, 4]]);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const r1 = await compiled(x);
      const r2 = await compiled(x);
      emit({ first: Array.from(r1.data), second: Array.from(r2.data) });
    `);
    expect(second).toEqual(first);
  });

  it('different inputs produce different outputs', () => {
    const { o1, o2 } = runGPUJSON(`
      const model = new Sequential(new Linear(4, 2));
      const x1 = tensor([[1, 0, 0, 0]]);
      const x2 = tensor([[0, 0, 0, 1]]);
      const compiled = compile(model, [x1], { target: WebGPUTarget() });
      emit({ o1: Array.from((await compiled(x1)).data), o2: Array.from((await compiled(x2)).data) });
    `);
    expect(o2).not.toEqual(o1);
  });

  it('zero input produces finite output', () => {
    const { gpu } = runGPUJSON(`
      const model = new Sequential(new Linear(4, 2));
      const x = tensor([[0, 0, 0, 0]]);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      emit({ gpu: Array.from((await compiled(x)).data) });
    `);
    expect(gpu.every(Number.isFinite), `non-finite in ${JSON.stringify(gpu)}`).toBe(true);
  });

  it('3-layer MLP executes correctly', () => {
    const { shape, gpu } = runGPUJSON(`
      const model = new Sequential(
        new Linear(8, 16), new ReLU(),
        new Linear(16, 8), new ReLU(),
        new Linear(8, 4),
      );
      const x = tensor([[1, 2, 3, 4, 5, 6, 7, 8]]);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      emit({ shape: out.shape, gpu: Array.from(out.data) });
    `);
    expect(shape).toEqual([1, 4]);
    expect(gpu.every(Number.isFinite), `non-finite in ${JSON.stringify(gpu)}`).toBe(true);
  });

  it('forward ignoring extra inputs matches eager (unused bindings)', () => {
    const out = runGPUJSON(`
      const model = { forward: (a, b, c) => sum(relu(a), 1, false) };
      const a = tensor([[1, -2, 3], [-4, 5, -6]]);
      const b = tensor([[9, 9, 9], [9, 9, 9]]);
      const c = tensor([[7, 7, 7], [7, 7, 7]]);
      const eager = model.forward(a, b, c);
      const compiled = compile(model, [a, b, c], { target: WebGPUTarget() });
      const out = await compiled(a, b, c);
      emit({ gpu: Array.from(out.data), eager: Array.from(eager.contiguous ? eager.contiguous().data : eager.data) });
    `);
    expect(out.gpu).toEqual([4, 5]);
    expect(maxErr(out.gpu, out.eager)).toBeLessThan(1e-4);
  });

  it('chained reduces with unused inputs match eager (multi-kernel buffers)', () => {
    const out = runGPUJSON(`
      const model = { forward: (a, b, c) => neg(mean(neg(mean(a, 1, false)), 1, false)) };
      const a = tensor([[0.5, -0.3], [0.2, 0.9]]);
      const b = tensor([[0.1, 0.4], [-0.2, 0.7]]);
      const c = tensor([[0.3, 0.1], [0.6, -0.5]]);
      const eager = model.forward(a, b, c);
      const compiled = compile(model, [a, b, c], { target: WebGPUTarget() });
      const out = await compiled(a, b, c);
      emit({ gpu: Array.from(out.data), eager: Array.from(eager.contiguous ? eager.contiguous().data : eager.data) });
    `);
    expect(maxErr(out.gpu, out.eager)).toBeLessThan(1e-4);
  });

  it('scheduled+autotuned conv2d+relu+maxpool matches CPU (multi-workgroup serialized)', () => {
    const out = runGPUJSON(`
      const mk = (seed, shape) => {
        const n = shape.reduce((a, b) => a * b, 1);
        const d = [];
        for (let i = 0; i < n; i++) d.push(Math.abs(Math.sin(i * 1.3 + seed)) * 2 - 0.5);
        const nest = (f, s) => s.length === 1 ? f.slice(0, s[0])
          : Array.from({ length: s[0] }, (_, i) => nest(f.slice(i * s.slice(1).reduce((a,b)=>a*b,1), (i+1) * s.slice(1).reduce((a,b)=>a*b,1)), s.slice(1)));
        return tensor(nest(d, shape));
      };
      const x = mk(1, [1, 2, 12, 12]);
      const w = mk(2, [3, 2, 3, 3]);
      const fwd = (a, b) => max_pool2d(relu(conv2d(a, b, null, [1, 1], [[0, 0], [0, 0]])), [2, 2], [2, 2]);
      const cpu = await compile({ forward: (a, b) => fwd(a, b) }, [x, w], { target: CPUTarget() });
      const ref = await cpu(x, w);
      const gc = compile({ forward: (a, b) => fwd(a, b) }, [x, w], { target: WebGPUTarget(), scheduling: { enabled: true, autotune: true, seed: 7 } });
      const out = await gc(x, w);
      emit({ gpu: Array.from(out.data), cpu: Array.from(ref.contiguous().data) });
    `);
    expect(relErr(out.gpu, out.cpu)).toBeLessThan(3e-3);
  });
});
