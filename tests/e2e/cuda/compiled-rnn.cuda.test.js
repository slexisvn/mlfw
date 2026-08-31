import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/compiler/support/target.js';
import { cudaDeps } from '../../_utils/cuda.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
function rng(s) { let x = s >>> 0; return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; }; }
function data(r, sh) {
  const n = sh.reduce((a, b) => a * b, 1); const f = [];
  for (let i = 0; i < n; i++) f.push(r() * 2 - 1);
  const nest = (fl, s) => s.length === 1 ? fl.slice(0, s[0]) : Array.from({ length: s[0] }, (_, i) => nest(fl.slice(i * fl.length / s[0], (i + 1) * fl.length / s[0]), s.slice(1)));
  return nest(f, sh);
}
async function run(target, fwd, x) {
  const c = compile({ forward: fwd }, [x], { target, scheduling: { enabled: true } });
  let r = c(x); if (r && r.then) r = await r;
  return flat(r);
}
async function err(model, sh) {
  model.eval();
  const x = tensor(data(rng(7), sh));
  const fwd = (a) => model.forward(a)[0];
  const cpu = await run(CPUTarget(), fwd, x);
  const gpu = await run(CUDATarget(), fwd, x);
  let e = 0, den = 0;
  for (let i = 0; i < cpu.length; i++) { e = Math.max(e, Math.abs(cpu[i] - gpu[i])); den = Math.max(den, Math.abs(cpu[i])); }
  return e / (1 + den);
}

describe.skipIf(!cudaDeps)('compiled scan RNN (LSTM/GRU) on CUDA matches CPU', () => {
  it('LSTM hidden=64 matches CPU', async () => {
    expect(await err(new nn.LSTM(8, 64, 1, false, true), [5, 2, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM hidden=128 long sequence matches CPU', async () => {
    expect(await err(new nn.LSTM(8, 128, 1, false, true), [20, 4, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM hidden=256 batch=8 matches CPU', async () => {
    expect(await err(new nn.LSTM(8, 256, 1, false, true), [10, 8, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM 2-layer hidden=64 matches CPU', async () => {
    expect(await err(new nn.LSTM(8, 64, 2, false, true), [10, 2, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM batchFirst hidden=64 matches CPU', async () => {
    expect(await err(new nn.LSTM(8, 64, 1, true, true), [2, 10, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('GRU hidden=128 2-layer matches CPU', async () => {
    expect(await err(new nn.GRU(8, 128, 2, false, true), [10, 2, 8])).toBeLessThan(2e-3);
  }, 60000);

  it('LSTM hidden=64 runs as a parallel persistent-RNN (carry in shared, not serialized)', () => {
    const model = new nn.LSTM(8, 64, 1, false, true);
    model.eval();
    const x = tensor(data(rng(7), [5, 2, 8]));
    const g = compile({ forward: (a) => model.forward(a)[0] }, [x], { target: CUDATarget(), scheduling: { enabled: true } });
    const mod = g.result().module;
    const md = mod.kernels.get(mod.listKernels()[0]).metadata;
    const total = md.blockDim.reduce((a, b) => a * b, 1) * md.gridDim.reduce((a, b) => a * b, 1);
    expect(total).toBeGreaterThan(1);
    expect(md.sharedMemBytes).toBeGreaterThan(0);
  });

  it('LSTM hidden=256 batch=16 matches CPU (serialized scan, large batch)', async () => {
    expect(await err(new nn.LSTM(8, 256, 1, false, true), [5, 16, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM input=32 hidden=256 batch=16 matches CPU', async () => {
    expect(await err(new nn.LSTM(32, 256, 1, false, true), [5, 16, 32])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM input=64 hidden=256 batch=16 matches CPU', async () => {
    expect(await err(new nn.LSTM(64, 256, 1, false, true), [5, 16, 64])).toBeLessThan(2e-3);
  }, 60000);
  it('LSTM hidden=256 batch=32 matches CPU (heavy serialized scan)', async () => {
    expect(await err(new nn.LSTM(8, 256, 1, false, true), [5, 32, 8])).toBeLessThan(2e-3);
  }, 60000);
  it('GRU hidden=256 batch=16 matches CPU', async () => {
    expect(await err(new nn.GRU(8, 256, 1, false, true), [5, 16, 8])).toBeLessThan(2e-3);
  }, 60000);

  it('large-batch serialized scan offloads oversized buffers to global scratch (no per-thread local-mem overflow)', () => {
    const model = new nn.LSTM(8, 256, 1, false, true);
    model.eval();
    const x = tensor(data(rng(7), [5, 16, 8]));
    const g = compile({ forward: (a) => model.forward(a)[0] }, [x], { target: CUDATarget(), scheduling: { enabled: true } });
    const k = g.result().module.kernels.get(g.result().module.listKernels()[0]);
    expect(k.metadata.blockDim.reduce((a, b) => a * b, 1)).toBe(1);
    expect(k.metadata.scratch.length).toBeGreaterThan(0);
    let localBytes = 0;
    const re = /\b(float|int|double)\s+\w+\[(\d+)\];/g;
    for (let m; (m = re.exec(k.source));) localBytes += Number(m[2]) * (m[1] === 'double' ? 8 : 4);
    expect(localBytes).toBeLessThan(512 * 1024);
  });
});
