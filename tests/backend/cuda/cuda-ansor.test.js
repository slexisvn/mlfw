import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { measureCudaKernel } from '#io/cuda_runtime';
import { cudaDeps } from './cuda-setup.js';
import { cu, checkCU } from '../../../src/io/node/cuda/ffi.js';
import { getDevice } from '../../../src/io/node/cuda/device.js';

const F32 = ScalarType.F32;
const rnd = (n, s) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.01 + s));

function peakFp32GFLOPs() {
  const dev = getDevice();
  const attr = (a) => { const p = [0]; checkCU('cuDeviceGetAttribute', cu.deviceGetAttribute(p, a, dev.dev)); return p[0]; };
  const sms = attr(16);
  const clockHz = attr(13) * 1000;
  return sms * 128 * clockHz * 2 / 1e9;
}

function autoScheduleMatmul(N) {
  const t = new TensorType([N, N], F32);
  const mk = () => buildFunction('mm', [t, t], [t], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
  const A = rnd(N * N, 1), B = rnd(N * N, 2);
  const cpu = new Float32Array(N * N), gpu = new Float32Array(N * N);
  compileGraph(mk(), CPUTarget(), { scheduling: { enabled: true } }).run('mm', A, B, cpu);
  const r = compileGraph(mk(), CUDATarget(), {
    scheduling: { enabled: true, autotune: true, seed: 7, hardwareMeasure: true, timeBudgetMs: 200000 },
  });
  return { r, A, B, cpu, gpu };
}

function maxRelErr(cpu, gpu) {
  let e = 0;
  for (let i = 0; i < cpu.length; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
  return e;
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function bestFraction(kernel, N, peakG, target, attempts = 6) {
  const bytes = [N * N * 4, N * N * 4, N * N * 4];
  let best = 0;
  for (let i = 0; i < attempts; i++) {
    await sleep(4000);
    const samples = measureCudaKernel(kernel, bytes, [], { warmup: 30, repeat: 60 });
    samples.sort((a, b) => a - b);
    const gflops = (2 * N * N * N) / (samples[0] * 1e6);
    best = Math.max(best, gflops / peakG);
    if (best >= target) break;
  }
  return best;
}

describe.skipIf(!cudaDeps)('CUDA Ansor-style auto-scheduled matmul', () => {
  it('auto-generates a register-blocked kernel reaching >=20% of FP32 peak', async () => {
    const peakG = peakFp32GFLOPs();
    const minFraction = 0.20;

    for (const N of [1024, 2048]) {
      const { r, A, B, cpu, gpu } = autoScheduleMatmul(N);
      await r.runAsync('mm', A, B, gpu);

      const err = maxRelErr(cpu, gpu);
      expect(err, `N=${N} maxRelErr vs CPU (${cudaDeps.arch})`).toBeLessThan(1e-3);

      const kernel = r.module.kernels.get('mm');
      expect(kernel.source.includes('__shared__') && kernel.source.includes('rb_acc'),
        `N=${N} kernel is register-blocked + shared-staged`).toBe(true);

      const fraction = await bestFraction(kernel, N, peakG, minFraction);
      expect(fraction, `N=${N} = ${(100 * fraction).toFixed(1)}% of ${peakG.toFixed(0)} GFLOP/s peak (${cudaDeps.arch})`)
        .toBeGreaterThanOrEqual(minFraction);
    }
  }, 300000);

  it('auto-scheduled matmul matches CPU on a non-power-of-two shape', async () => {
    const N = 500;
    const t = new TensorType([N, N], F32);
    const mk = () => buildFunction('mm', [t, t], [t], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
    const A = rnd(N * N, 3), B = rnd(N * N, 4);
    const cpu = new Float32Array(N * N), gpu = new Float32Array(N * N);
    compileGraph(mk(), CPUTarget(), { scheduling: { enabled: true } }).run('mm', A, B, cpu);
    const r = compileGraph(mk(), CUDATarget(), {
      scheduling: { enabled: true, autotune: true, seed: 11, hardwareMeasure: true, timeBudgetMs: 120000 },
    });
    await r.runAsync('mm', A, B, gpu);
    expect(maxRelErr(cpu, gpu), `N=${N} maxRelErr (${cudaDeps.arch})`).toBeLessThan(1e-3);
  }, 200000);
});
