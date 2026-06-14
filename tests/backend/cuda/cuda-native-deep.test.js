import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function data(rng, s, lo, hi) {
  const n = s.reduce((a, b) => a * b, 1);
  const f = [];
  for (let i = 0; i < n; i++) f.push(lo + (hi - lo) * rng());
  const nest = (fl, sh) => sh.length === 1 ? fl.slice(0, sh[0]) : Array.from({ length: sh[0] }, (_, i) => nest(fl.slice(i * fl.length / sh[0], (i + 1) * fl.length / sh[0]), sh.slice(1)));
  return nest(f, s);
}

async function deepMlpErr(width, batch) {
  const rng = mulberry32(4000 + width);
  const l1 = new nn.Linear(width, width); l1.eval();
  const l2 = new nn.Linear(width, width); l2.eval();
  const l3 = new nn.Linear(width, Math.max(2, width >> 1)); l3.eval();
  const fwd = (x) => l3.forward(l2.forward(l1.forward(x).relu()).relu());
  const x = tensor(data(rng, [batch, width], -1, 1));

  const cpu = flat(await compile({ forward: fwd }, [x], { target: CPUTarget() })(x));
  const cf = compile({ forward: fwd }, [x], { target: CUDATarget(), verify: false, scheduling: { enabled: true } });
  const plan = cf.result().module.executionPlan;
  let g = cf(x); if (g && g.then) g = await g; g = flat(g);

  let e = 0;
  for (let i = 0; i < cpu.length; i++) e = Math.max(e, Math.abs(cpu[i] - g[i]) / (1 + Math.abs(cpu[i])));
  return { err: e, split: !!plan, kernels: cf.result().module.listKernels().length };
}

describe.skipIf(!cudaDeps)('CUDA native multi-kernel deep models match CPU on real GPU', () => {
  for (const [width, batch] of [[8, 1], [64, 8], [256, 32], [512, 64], [1024, 128]]) {
    it(`3-layer MLP width=${width} batch=${batch} native CUDA == CPU`, async () => {
      const r = await deepMlpErr(width, batch);
      expect(r.split, 'matmul chain split into multiple kernels').toBe(true);
      expect(r.kernels, 'multiple kernels').toBeGreaterThan(1);
      expect(r.err, `width=${width} maxRelErr (${cudaDeps.arch})`).toBeLessThan(2e-3);
    }, 120000);
  }
});
