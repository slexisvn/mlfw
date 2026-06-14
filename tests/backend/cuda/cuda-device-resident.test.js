import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { buildGpt2 } from '../../shared/gpt2_model.js';
import { getBackend } from '../../../src/compiler/runtime/backend_registry.js';
import { cudaDeps } from './cuda-setup.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
const maxRelErr = (a, b) => {
  let e = 0;
  for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return e;
};

async function runPerStep(cf, input) {
  const backend = getBackend('cuda');
  const saved = backend.runPlan;
  backend.runPlan = undefined;
  try {
    return flat(await cf(input));
  } finally {
    backend.runPlan = saved;
  }
}

function planOf(cf) {
  const mod = cf.result().module;
  let cublasSteps = 0;
  for (const name of mod.listKernels()) {
    const k = mod.kernels.get(name);
    if (k && k.metadata && k.metadata.cublas) cublasSteps++;
  }
  return { plan: mod.executionPlan, cublasSteps };
}

describe.skipIf(!cudaDeps)('CUDA device-resident multi-kernel executor', () => {
  it('Linear chain keeps intermediates on device and matches CPU + per-step', async () => {
    const M = 12, K = 80, H = 64, N = 48;
    const x = tensor(Array.from({ length: M }, (_, i) =>
      Array.from({ length: K }, (_, j) => Math.sin(i * 0.11 + j * 0.019))));
    const l1 = new nn.Linear(K, H); l1.eval();
    const l2 = new nn.Linear(H, N); l2.eval();
    const fwd = (inp) => l2.forward(l1.forward(inp).relu()).relu();

    const cpu = flat(await compile({ forward: fwd }, [x], { target: CPUTarget() })(x));
    const cf = compile({ forward: fwd }, [x], { target: CUDATarget(), matmulBackend: 'cublas' });

    const { plan, cublasSteps } = planOf(cf);
    expect(plan, 'execution plan present').toBeTruthy();
    expect(plan.intermediates.length, 'plan has device-resident intermediates').toBeGreaterThan(0);
    expect(cublasSteps, 'at least two cublas matmul steps').toBeGreaterThanOrEqual(2);

    const deviceResident = flat(await cf(x));
    const perStep = await runPerStep(cf, x);

    expect(deviceResident.length).toBe(cpu.length);
    expect(maxRelErr(cpu, deviceResident), 'device-resident vs CPU').toBeLessThan(2e-3);
    expect(maxRelErr(perStep, deviceResident), 'device-resident vs per-step').toBeLessThan(1e-5);
  }, 120000);

  it('small GPT-2 device-resident plan matches CPU + per-step', async () => {
    const dims = { V: 96, D: 64, H: 4, FF: 128, L: 2, SEQ: 12 };
    const { fwd, ids } = buildGpt2(nn, tensor, dims);

    const cpu = flat(await compile({ forward: fwd }, [ids], { target: CPUTarget() })(ids));
    const cf = compile({ forward: fwd }, [ids], { target: CUDATarget(), matmulBackend: 'cublas' });

    const { plan, cublasSteps } = planOf(cf);
    expect(plan.intermediates.length, 'gpt2 device-resident intermediates').toBeGreaterThan(0);
    expect(cublasSteps, 'gpt2 cublas steps').toBeGreaterThanOrEqual(2);

    const deviceResident = flat(await cf(ids));
    const perStep = await runPerStep(cf, ids);

    expect(maxRelErr(cpu, deviceResident), `gpt2 device-resident vs CPU (${cudaDeps.arch})`).toBeLessThan(2e-3);
    expect(maxRelErr(perStep, deviceResident), 'gpt2 device-resident vs per-step').toBeLessThan(1e-5);
  }, 180000);
});
