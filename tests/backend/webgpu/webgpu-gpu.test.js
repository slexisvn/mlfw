import { describe, it, expect } from 'vitest';
import { execFileSync } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { tmpdir } from 'os';

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

let _tmpCounter = 0;

function toFileURL(p) {
  return 'file:///' + p.replace(/\\/g, '/');
}

function runGPUTest(testBody) {
  const tmpFile = join(tmpdir(), `_webgpu_test_${Date.now()}_${_tmpCounter++}.mjs`);
  const indexURL = toFileURL(resolve(PROJECT_ROOT, 'src/index.js'));
  const runtimeURL = toFileURL(resolve(PROJECT_ROOT, 'src/compiler/runtime/webgpu_runtime.js'));
  const script = `
import { tensor, Linear, Sequential, ReLU, Sigmoid, Tanh, compile, WebGPUTarget, relu, sum, mean, neg, add, mul } from '${indexURL}';
import { resetDevice } from '${runtimeURL}';
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
    const out = runGPUTest(`
      const model = new Sequential(new Linear(4, 2));
      const x = tensor([[1, 2, 3, 4]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      if (out.shape[0] !== 1 || out.shape[1] !== 2) throw new Error('bad shape: ' + JSON.stringify(out.shape));
      let d = 0;
      for (let i = 0; i < eager.data.length; i++) d = Math.max(d, Math.abs(eager.data[i] - out.data[i]));
      if (d > 0.001) throw new Error('mismatch: ' + d);
      process.stdout.write(JSON.stringify(out.shape));
    `);
    expect(JSON.parse(out)).toEqual([1, 2]);
  });

  it('Linear + ReLU clamps negatives', () => {
    const out = runGPUTest(`
      const model = new Sequential(new Linear(4, 4), new ReLU());
      const x = tensor([[1, -1, 2, -2]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      for (let i = 0; i < out.data.length; i++) {
        if (out.data[i] < 0) throw new Error('negative: ' + out.data[i]);
      }
      let d = 0;
      for (let i = 0; i < eager.data.length; i++) d = Math.max(d, Math.abs(eager.data[i] - out.data[i]));
      if (d > 0.001) throw new Error('mismatch: ' + d);
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });

  it('Linear + Sigmoid bounds [0,1]', () => {
    const out = runGPUTest(`
      const model = new Sequential(new Linear(3, 3), new Sigmoid());
      const x = tensor([[10, -10, 0]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      for (let i = 0; i < out.data.length; i++) {
        if (out.data[i] < 0 || out.data[i] > 1) throw new Error('out of range: ' + out.data[i]);
      }
      let d = 0;
      for (let i = 0; i < eager.data.length; i++) d = Math.max(d, Math.abs(eager.data[i] - out.data[i]));
      if (d > 0.001) throw new Error('mismatch: ' + d);
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });

  it('Linear + Tanh bounds [-1,1]', () => {
    const out = runGPUTest(`
      const model = new Sequential(new Linear(3, 3), new Tanh());
      const x = tensor([[5, -5, 0]]);
      const eager = model.forward(x);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      for (let i = 0; i < out.data.length; i++) {
        if (out.data[i] < -1 || out.data[i] > 1) throw new Error('out of range: ' + out.data[i]);
      }
      let d = 0;
      for (let i = 0; i < eager.data.length; i++) d = Math.max(d, Math.abs(eager.data[i] - out.data[i]));
      if (d > 0.001) throw new Error('mismatch: ' + d);
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });

  it('deterministic across repeated runs', () => {
    const out = runGPUTest(`
      const model = new Sequential(new Linear(4, 2));
      const x = tensor([[1, 2, 3, 4]]);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const r1 = await compiled(x);
      const r2 = await compiled(x);
      let d = 0;
      for (let i = 0; i < r1.data.length; i++) d = Math.max(d, Math.abs(r1.data[i] - r2.data[i]));
      if (d > 1e-6) throw new Error('non-deterministic: ' + d);
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });

  it('different inputs produce different outputs', () => {
    const out = runGPUTest(`
      const model = new Sequential(new Linear(4, 2));
      const x1 = tensor([[1, 0, 0, 0]]);
      const x2 = tensor([[0, 0, 0, 1]]);
      const compiled = compile(model, [x1], { target: WebGPUTarget() });
      const o1 = await compiled(x1);
      const o2 = await compiled(x2);
      const diff = Math.abs(o1.data[0] - o2.data[0]) + Math.abs(o1.data[1] - o2.data[1]);
      if (diff === 0) throw new Error('identical outputs');
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });

  it('zero input produces finite output', () => {
    const out = runGPUTest(`
      const model = new Sequential(new Linear(4, 2));
      const x = tensor([[0, 0, 0, 0]]);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      for (let i = 0; i < out.data.length; i++) {
        if (!Number.isFinite(out.data[i])) throw new Error('non-finite: ' + out.data[i]);
      }
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });

  it('3-layer MLP executes correctly', () => {
    const out = runGPUTest(`
      const model = new Sequential(
        new Linear(8, 16), new ReLU(),
        new Linear(16, 8), new ReLU(),
        new Linear(8, 4),
      );
      const x = tensor([[1, 2, 3, 4, 5, 6, 7, 8]]);
      const compiled = compile(model, [x], { target: WebGPUTarget() });
      const out = await compiled(x);
      if (out.shape[0] !== 1 || out.shape[1] !== 4) throw new Error('bad shape');
      for (let i = 0; i < out.data.length; i++) {
        if (!Number.isFinite(out.data[i])) throw new Error('non-finite');
      }
      process.stdout.write(JSON.stringify(out.shape));
    `);
    expect(JSON.parse(out)).toEqual([1, 4]);
  });

  // Regression: inputs that the graph never reads still get a @binding slot in
  // the generated WGSL. With layout:'auto' those bindings were pruned from the
  // pipeline's bind group layout, so the runtime's bind group (which supplies a
  // buffer per declared binding) referenced indices absent from the layout —
  // CreateBindGroup failed validation and the output buffer stayed zero. The
  // runtime now builds an explicit bind group layout from the kernel bindings.
  it('forward ignoring extra inputs matches eager (unused bindings)', () => {
    const out = runGPUTest(`
      const model = { forward: (a, b, c) => sum(relu(a), 1, false) };
      const a = tensor([[1, -2, 3], [-4, 5, -6]]);
      const b = tensor([[9, 9, 9], [9, 9, 9]]);
      const c = tensor([[7, 7, 7], [7, 7, 7]]);
      const eager = model.forward(a, b, c);
      const compiled = compile(model, [a, b, c], { target: WebGPUTarget() });
      const out = await compiled(a, b, c);
      const e = Array.from(eager.contiguous ? eager.contiguous().data : eager.data);
      let d = 0;
      for (let i = 0; i < e.length; i++) d = Math.max(d, Math.abs(e[i] - out.data[i]));
      if (d > 1e-4) throw new Error('mismatch: ' + d + ' eager=' + JSON.stringify(e) + ' gpu=' + JSON.stringify(Array.from(out.data)));
      process.stdout.write(JSON.stringify(Array.from(out.data)));
    `);
    expect(JSON.parse(out)).toEqual([4, 5]);
  });

  it('chained reduces with unused inputs match eager (multi-kernel buffers)', () => {
    const out = runGPUTest(`
      const model = { forward: (a, b, c) => neg(mean(neg(mean(a, 1, false)), 1, false)) };
      const a = tensor([[0.5, -0.3], [0.2, 0.9]]);
      const b = tensor([[0.1, 0.4], [-0.2, 0.7]]);
      const c = tensor([[0.3, 0.1], [0.6, -0.5]]);
      const eager = model.forward(a, b, c);
      const compiled = compile(model, [a, b, c], { target: WebGPUTarget() });
      const out = await compiled(a, b, c);
      const e = Array.from(eager.contiguous ? eager.contiguous().data : eager.data);
      let d = 0;
      for (let i = 0; i < e.length; i++) d = Math.max(d, Math.abs(e[i] - out.data[i]));
      if (d > 1e-4) throw new Error('mismatch: ' + d);
      process.stdout.write('OK');
    `);
    expect(out).toBe('OK');
  });
});
