import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { buildGpt2 } from '../../shared/gpt2_model.js';
import { getBackend } from '../../../src/compiler/runtime/backend_registry.js';
import { cudaDeps } from './cuda-setup.js';

const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);
const maxRelErr = (a, b) => {
  let e = 0;
  for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return e;
};

describe.skipIf(!cudaDeps)('CUDA compiled multi-kernel & cuBLAS plans', () => {
  describe('matmulBackend: cublas', () => {
    const F32 = ScalarType.F32;
    const rnd = (n, s) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.01 + s));

    async function diff(M, K, N) {
      const mk = () => buildFunction('mm', [new TensorType([M, K], F32), new TensorType([K, N], F32)], [new TensorType([M, N], F32)], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
      const A = rnd(M * K, 1), B = rnd(K * N, 2);
      const cpu = new Float32Array(M * N), gpu = new Float32Array(M * N);
      compileGraph(mk(), CPUTarget()).run('mm', A, B, cpu);
      const r = compileGraph(mk(), CUDATarget(), { matmulBackend: 'cublas' });
      const usedCublas = !!r.module.kernels.get('mm').metadata.cublas;
      await r.runAsync('mm', A, B, gpu);
      let e = 0;
      for (let i = 0; i < M * N; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
      return { e, usedCublas };
    }

    it('cublas mode dispatches matmul and matches CPU across shapes', async () => {
      for (const [M, K, N] of [[16, 24, 16], [128, 256, 64], [7, 13, 5], [256, 256, 256]]) {
        const { e, usedCublas } = await diff(M, K, N);
        expect(usedCublas, `${M}x${K}x${N} used cublas`).toBe(true);
        expect(e, `${M}x${K}x${N} maxRelErr`).toBeLessThan(1e-2);
      }
    }, 60000);

    it('native and cublas modes coexist in one process', async () => {
      const t = new TensorType([64, 64], F32);
      const mk = () => buildFunction('mm', [t, t], [t], (b, a) => b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]));
      const A = rnd(4096, 3), B = rnd(4096, 4);
      const cpu = new Float32Array(4096), gpu = new Float32Array(4096);
      compileGraph(mk(), CPUTarget(), { scheduling: { enabled: true } }).run('mm', A, B, cpu);
      const r = compileGraph(mk(), CUDATarget(), { scheduling: { enabled: true } });
      await r.runAsync('mm', A, B, gpu);
      let e = 0;
      for (let i = 0; i < 4096; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
      expect(e).toBeLessThan(1e-2);
    }, 60000);
  });

  describe('multi-kernel cublas split', () => {
    function inspect(compiled) {
      const mod = compiled.result().module;
      let cublasSteps = 0;
      for (const name of mod.listKernels()) {
        const k = mod.kernels.get(name);
        if (k && k.metadata && k.metadata.cublas) cublasSteps++;
      }
      return { plan: mod.executionPlan, kernelCount: mod.listKernels().length, cublasSteps };
    }

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
      expect(maxRelErr(cpu, gpu), 'linear+relu cublas-vs-cpu').toBeLessThan(1e-2);
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
      expect(maxRelErr(cpu, gpu), `gpt2 cublas-vs-cpu (${cudaDeps.arch})`).toBeLessThan(1e-2);
    }, 180000);
  });

  describe('native multi-kernel deep models', () => {
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

    for (const [width, batch] of [[8, 1], [64, 8], [256, 32], [512, 64], [1024, 128]]) {
      it(`3-layer MLP width=${width} batch=${batch} native CUDA == CPU`, async () => {
        const r = await deepMlpErr(width, batch);
        expect(r.split, 'matmul chain split into multiple kernels').toBe(true);
        expect(r.kernels, 'multiple kernels').toBeGreaterThan(1);
        expect(r.err, `width=${width} maxRelErr (${cudaDeps.arch})`).toBeLessThan(1e-2);
      }, 120000);
    }
  });

  describe('device-resident multi-kernel executor', () => {
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
      expect(maxRelErr(cpu, deviceResident), 'device-resident vs CPU').toBeLessThan(1e-2);
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

      expect(maxRelErr(cpu, deviceResident), `gpt2 device-resident vs CPU (${cudaDeps.arch})`).toBeLessThan(1e-2);
      expect(maxRelErr(perStep, deviceResident), 'gpt2 device-resident vs per-step').toBeLessThan(1e-5);
    }, 180000);
  });
});
