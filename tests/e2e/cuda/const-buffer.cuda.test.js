import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { setH2DObserver } from '../../../src/runtime/node/cuda/runtime_api.js';
import { releaseCudaMemory } from '../../../src/runtime/backend_registry.js';
import { buildGpt2 } from '../../_utils/gpt2_model.js';
import { cudaDeps } from '../../_utils/cuda.js';
import { mulberry32 } from '../../_utils/rng.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
const maxRelErr = (a, b) => {
  let e = 0;
  for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return e;
};

function constBufferCount(compiled) {
  const result = compiled.result();
  let n = 0;
  for (const name of result.listKernels()) {
    const cbs = result.module.getKernelMetadata(name).constBuffers;
    if (cbs) n += cbs.length;
  }
  return n;
}

async function ready(compiled) {
  if (compiled._ready) await compiled._ready;
  return compiled;
}

describe.skipIf(!cudaDeps)('CUDA folded weights bound as constant buffers', () => {
  it('4x Linear(128,128) at batch 64 matches the unfolded compile', async () => {
    const W = 128, B = 64;
    const rng = mulberry32(7331);
    const layers = [];
    for (let i = 0; i < 4; i++) {
      const l = new nn.Linear(W, W, false);
      for (let k = 0; k < l.weight.data.length; k++) l.weight.data[k] = (rng() - 0.5) * 0.2;
      l.eval();
      layers.push(l);
    }
    const fwd = (x) => layers.reduce((h, l) => l.forward(h), x);

    const rows = [];
    for (let i = 0; i < B; i++) {
      const r = [];
      for (let d = 0; d < W; d++) r.push((rng() - 0.5) * 2);
      rows.push(r);
    }
    const x = tensor(rows);

    const cpu = flat(await (await ready(compile({ forward: fwd }, [x], { target: CPUTarget() })))(x));
    const plain = await ready(compile({ forward: fwd }, [x], { target: CUDATarget() }));
    const folded = await ready(compile({ forward: fwd }, [x], { target: CUDATarget(), foldWeights: true }));

    expect(constBufferCount(plain), 'unfolded compile links no constants').toBe(0);
    expect(constBufferCount(folded), 'every weight linked as a constant buffer').toBe(4);
    expect(folded.source().length, 'folded source stays near the unfolded size')
      .toBeLessThan(plain.source().length + 4 * W * W);

    const foldedOut = flat(await folded(x));
    expect(foldedOut.length).toBe(cpu.length);
    expect(maxRelErr(cpu, foldedOut), `4x Linear(128,128) folded-vs-CPU (${cudaDeps.arch})`).toBeLessThan(1e-6);
    expect(maxRelErr(flat(await plain(x)), foldedOut), 'folded vs unfolded CUDA').toBeLessThan(1e-7);
  }, 300000);

  it('small GPT-2 decoder matches the unfolded compile', async () => {
    const dims = { V: 64, D: 32, H: 4, FF: 64, L: 2, SEQ: 8 };
    const { fwd, ids } = buildGpt2(nn, tensor, dims);

    const cpu = flat(await (await ready(compile({ forward: fwd }, [ids], { target: CPUTarget() })))(ids));
    const plain = await ready(compile({ forward: fwd }, [ids], { target: CUDATarget() }));
    const folded = await ready(compile({ forward: fwd }, [ids], { target: CUDATarget(), foldWeights: true }));

    expect(constBufferCount(folded), 'gpt2 links constants').toBeGreaterThan(0);

    const foldedOut = flat(await folded(ids));
    expect(foldedOut.length).toBe(cpu.length);
    expect(maxRelErr(cpu, foldedOut), `gpt2 folded-vs-CPU (${cudaDeps.arch})`).toBeLessThan(1e-5);
    expect(maxRelErr(flat(await plain(ids)), foldedOut), 'gpt2 folded vs unfolded CUDA').toBeLessThan(1e-7);
  }, 300000);

  it('uploads each constant buffer once across repeated plan execution', async () => {
    const W = 128;
    const rng = mulberry32(99);
    const layers = [];
    for (let i = 0; i < 2; i++) {
      const l = new nn.Linear(W, W, false);
      for (let k = 0; k < l.weight.data.length; k++) l.weight.data[k] = (rng() - 0.5) * 0.2;
      l.eval();
      layers.push(l);
    }
    const fwd = (x) => layers.reduce((h, l) => l.forward(h), x);

    const rows = [];
    for (let i = 0; i < 8; i++) {
      const r = [];
      for (let d = 0; d < W; d++) r.push(rng());
      rows.push(r);
    }
    const x = tensor(rows);

    await releaseCudaMemory();
    const folded = await ready(compile({ forward: fwd }, [x], { target: CUDATarget(), foldWeights: true }));
    expect(constBufferCount(folded)).toBe(2);
    expect(folded.result().module.executionPlan, 'multi-kernel plan present').toBeTruthy();

    const measure = async () => {
      let bytes = 0;
      setH2DObserver((n) => { bytes += n; });
      try { await folded(x); } finally { setH2DObserver(null); }
      return bytes;
    };

    const first = await measure();
    const second = await measure();
    const third = await measure();

    const weightBytes = 2 * W * W * 4;
    expect(first - second, 'the first run alone uploads the constants').toBeGreaterThanOrEqual(weightBytes);
    expect(third, 'later runs upload the same bytes as the second').toBe(second);
    expect(second, 'later runs never re-upload the constants').toBeLessThan(weightBytes);
  }, 300000);

  it('binds constant buffers on the single-kernel launch path', async () => {
    const W = 32;
    const rng = mulberry32(2024);
    const l = new nn.Linear(W, W, false);
    for (let k = 0; k < l.weight.data.length; k++) l.weight.data[k] = (rng() - 0.5) * 0.4;
    l.eval();
    const fwd = (x) => l.forward(x).relu();

    const rows = [];
    for (let i = 0; i < 4; i++) {
      const r = [];
      for (let d = 0; d < W; d++) r.push(rng() - 0.5);
      rows.push(r);
    }
    const x = tensor(rows);

    const cpu = flat(await (await ready(compile({ forward: fwd }, [x], { target: CPUTarget() })))(x));
    const folded = await ready(compile({ forward: fwd }, [x], { target: CUDATarget(), foldWeights: true }));

    expect(constBufferCount(folded)).toBe(1);
    expect(folded.result().module.executionPlan, 'single kernel, no plan').toBeFalsy();

    const out = flat(await folded(x));
    expect(maxRelErr(cpu, out), 'single-kernel folded-vs-CPU').toBeLessThan(1e-6);
  }, 300000);
});
