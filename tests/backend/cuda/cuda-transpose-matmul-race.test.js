import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as ops from '../../../src/tensor/ops/ops.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { buildGpt2 } from '../../shared/gpt2_model.js';
import { cudaDeps } from './cuda-setup.js';

const flat = v => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
function rng(s) { let x = s >>> 0; return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; }; }
function data(r, sh) {
  const n = sh.reduce((a, b) => a * b, 1); const f = [];
  for (let i = 0; i < n; i++) f.push(r() * 2 - 1);
  const nest = (fl, s) => s.length === 1 ? fl.slice(0, s[0]) : Array.from({ length: s[0] }, (_, i) => nest(fl.slice(i * fl.length / s[0], (i + 1) * fl.length / s[0]), s.slice(1)));
  return nest(f, sh);
}
async function err(fwd, x) {
  const cpu = flat(await (async () => { const c = compile({ forward: fwd }, [x], { target: CPUTarget() }); let r = c(x); if (r && r.then) r = await r; return r; })());
  const g = compile({ forward: fwd }, [x], { target: CUDATarget(), verify: false, scheduling: { enabled: true } });
  let r = g(x); if (r && r.then) r = await r; const gpu = flat(r);
  let e = 0; for (let i = 0; i < cpu.length; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
  return e;
}
const tl = t => t.transpose(t.ndim - 2, t.ndim - 1);

describe.skipIf(!cudaDeps)('CUDA transpose/permute -> matmul cross-thread race', () => {
  it('4D batched matmul(x, transpose(x)) matches CPU', async () => {
    expect(await err(a => ops.matmul(a, tl(a)), tensor(data(rng(7), [1, 4, 32, 32])))).toBeLessThan(2e-3);
  }, 60000);
  it('4D batched matmul(x, transpose(x)) batch>1 matches CPU', async () => {
    expect(await err(a => ops.matmul(a, tl(a)), tensor(data(rng(9), [2, 4, 32, 32])))).toBeLessThan(2e-3);
  }, 60000);
  it('standalone reshape+permute matches CPU', async () => {
    expect(await err(a => a.reshape([1, 32, 4, 32]).permute([0, 2, 1, 3]), tensor(data(rng(3), [1, 32, 128])))).toBeLessThan(2e-3);
  }, 60000);
  it('MultiheadAttention (small dims D128 H4 S32) matches CPU', async () => {
    const at = new nn.MultiheadAttention(128, 4, 0, true, null, null, true); at.eval();
    const fwd = a => { const o = at.forward(a, a, a); return Array.isArray(o) ? o[0] : o; };
    expect(await err(fwd, tensor(data(rng(7), [1, 32, 128])))).toBeLessThan(2e-3);
  }, 60000);
  it('GPT-2 (tiny D128 L2 S32) end-to-end matches CPU', async () => {
    const { fwd, ids } = buildGpt2(nn, tensor, { V: 256, D: 128, H: 4, FF: 512, L: 2, SEQ: 32 });
    expect(await err(fwd, ids)).toBeLessThan(2e-2);
  }, 60000);

  it('ordinary batched matmul stays fully parallel (not serialized)', () => {
    const g = compile({ forward: a => ops.matmul(a, tensor(data(rng(2), [1, 4, 32, 32]))) },
      [tensor(data(rng(7), [1, 4, 32, 32]))], { target: CUDATarget(), scheduling: { enabled: true } });
    const mod = g.result().module;
    const md = mod.kernels.get(mod.listKernels()[0]).metadata;
    const total = md.blockDim.reduce((a, b) => a * b, 1) * md.gridDim.reduce((a, b) => a * b, 1);
    expect(total).toBeGreaterThan(1);
  });

  it('matmul(x, transpose(x)) is inlined and stays fully parallel (not serialized)', () => {
    const g = compile({ forward: a => ops.matmul(a, tl(a)) },
      [tensor(data(rng(7), [1, 4, 32, 32]))], { target: CUDATarget(), scheduling: { enabled: true } });
    const mod = g.result().module;
    for (const name of mod.listKernels()) {
      const md = mod.kernels.get(name).metadata;
      const total = md.blockDim.reduce((a, b) => a * b, 1) * md.gridDim.reduce((a, b) => a * b, 1);
      expect(total).toBeGreaterThan(1);
    }
  });
});
