import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { buildGpt2 } from '../../shared/gpt2_model.js';
import { cudaDeps } from './cuda-setup.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

async function maxRelErr(fwd, inputs) {
  const cpu = flat(await compile({ forward: fwd }, inputs, { target: CPUTarget() })(...inputs));
  const gpu = flat(await compile({ forward: fwd }, inputs, { target: CUDATarget() })(...inputs));
  expect(gpu.length).toBe(cpu.length);
  let e = 0;
  for (let i = 0; i < cpu.length; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
  return e;
}

describe.skipIf(!cudaDeps)('CUDA executes full models matching CPU on real GPU', () => {
  it('GPT-2 decoder compiled on CUDA matches CPU', async () => {
    const { fwd, ids } = buildGpt2(nn, tensor, { V: 64, D: 48, H: 3, FF: 96, L: 2, SEQ: 8 });
    const e = await maxRelErr(fwd, [ids]);
    expect(e, `gpt2 cuda-vs-cpu maxRelErr (${cudaDeps.arch})`).toBeLessThan(2e-3);
  }, 180000);
});
