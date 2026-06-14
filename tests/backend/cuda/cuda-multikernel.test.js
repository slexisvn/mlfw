import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { buildGpt2 } from '../../shared/gpt2_model.js';
import { cudaDeps } from './cuda-setup.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

function inspect(compiled) {
  const mod = compiled.result().module;
  let cublasSteps = 0;
  for (const name of mod.listKernels()) {
    const k = mod.kernels.get(name);
    if (k && k.metadata && k.metadata.cublas) cublasSteps++;
  }
  return { plan: mod.executionPlan, kernelCount: mod.listKernels().length, cublasSteps };
}

function maxRelErr(a, b) {
  let e = 0;
  for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return e;
}

describe.skipIf(!cudaDeps)('CUDA multi-kernel cublas split', () => {
  it('Linear+ReLU splits into multiple kernels with a cublas step and matches CPU', async () => {
    const M = 8, K = 64, N = 96;
    const x = tensor(Array.from({ length: M }, (_, i) =>
      Array.from({ length: K }, (_, j) => Math.sin(i * 0.13 + j * 0.017))));
    const lin = new nn.Linear(K, N);
    lin.eval();
    const fwd = (inp) => lin.forward(inp).relu();

    const cpu = flat(await compile({ forward: fwd }, [x], { target: CPUTarget() })(x));

    const gpuCompiled = compile({ forward: fwd }, [x], { target: CUDATarget(), matmulBackend: 'cublas' });
    const info = inspect(gpuCompiled);
    expect(info.plan, 'execution plan present').toBeTruthy();
    expect(info.kernelCount, 'multiple kernels').toBeGreaterThan(1);
    expect(info.cublasSteps, 'at least one cublas step').toBeGreaterThanOrEqual(1);

    const gpu = flat(await gpuCompiled(x));
    expect(gpu.length).toBe(cpu.length);
    expect(maxRelErr(cpu, gpu), 'linear+relu cublas-vs-cpu').toBeLessThan(2e-3);
  }, 120000);

  it('small GPT-2 routes matmuls through cublas and matches CPU', async () => {
    const dims = { V: 64, D: 48, H: 3, FF: 96, L: 2, SEQ: 8 };
    const { fwd, ids } = buildGpt2(nn, tensor, dims);
    const cpu = flat(await compile({ forward: fwd }, [ids], { target: CPUTarget() })(ids));

    const gpuCompiled = compile({ forward: fwd }, [ids], { target: CUDATarget(), matmulBackend: 'cublas' });
    const info = inspect(gpuCompiled);
    expect(info.plan, 'execution plan present').toBeTruthy();
    expect(info.kernelCount, 'multiple kernels').toBeGreaterThan(1);
    expect(info.cublasSteps, 'at least one cublas step').toBeGreaterThanOrEqual(1);

    const gpu = flat(await gpuCompiled(ids));
    expect(gpu.length).toBe(cpu.length);
    expect(maxRelErr(cpu, gpu), `gpt2 cublas-vs-cpu (${cudaDeps.arch})`).toBeLessThan(2e-3);
  }, 180000);
});
