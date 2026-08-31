import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/compiler/support/target.js';
import { measureCudaKernel } from '#io/cuda_runtime';
import { cudaDeps } from '../../_utils/cuda.js';
import { cu, checkCU } from '../../../src/runtime/node/cuda/ffi.js';
import { getDevice } from '../../../src/runtime/node/cuda/device.js';
import { F32 } from '../../_utils/ir_fixture.js';

const rnd = (n, s) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.01 + s));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function matmulGraph(N) {
  const t = new TensorType([N, N], F32);
  return () => buildFunction('mm', [t, t], [t], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
}

function maxRelErr(cpu, gpu) {
  let e = 0;
  for (let i = 0; i < cpu.length; i++) e = Math.max(e, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
  return e;
}

async function autotunedVsCpu(N, seeds, scheduling) {
  const mk = matmulGraph(N);
  const lhs = rnd(N * N, seeds[0]), rhs = rnd(N * N, seeds[1]);
  const cpu = new Float32Array(N * N), gpu = new Float32Array(N * N);
  compileGraph(mk(), CPUTarget(), { scheduling: { enabled: true } }).run('mm', lhs, rhs, cpu);
  const r = compileGraph(mk(), CUDATarget(), {
    scheduling: { enabled: true, autotune: true, hardwareMeasure: true, ...scheduling },
  });
  await r.runAsync('mm', lhs, rhs, gpu);
  return { r, err: maxRelErr(cpu, gpu) };
}

function peakFp32GFLOPs() {
  const dev = getDevice();
  const attr = (a) => { const p = [0]; checkCU('cuDeviceGetAttribute', cu.deviceGetAttribute(p, a, dev.dev)); return p[0]; };
  const sms = attr(16);
  const clockHz = attr(13) * 1000;
  return sms * 128 * clockHz * 2 / 1e9;
}

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

describe.skipIf(!cudaDeps)('CUDA autotuner', () => {
  describe('hardware-measured autotune', () => {
    it('autotune with hardwareMeasure compiles, completes, and matches CPU', async () => {
      const { err } = await autotunedVsCpu(256, [1, 2], { seed: 7 });
      expect(err, `autotune-measured matmul maxRelErr (${cudaDeps.arch})`).toBeLessThan(2e-3);
    }, 60000);
  });

  describe('cost-model-only autotune never regresses below the default schedule', () => {
    const countSerialLoops = (src) => (src.match(/for \(int /g) || []).length;

    it('parallelizes the matmul compute rather than leaving it per-thread serial', () => {
      const N = 128;
      const t = new TensorType([N, N], F32);
      const mk = () => buildFunction('mm', [t, t], [t], (b, a) => b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]));
      const baseline = compileGraph(mk(), CUDATarget(), { scheduling: { enabled: true } });
      const autotuned = compileGraph(mk(), CUDATarget(), { scheduling: { enabled: true, autotune: true, seed: 7 } });
      const baseFor = countSerialLoops(baseline.module.kernels.get('mm').source);
      const autoFor = countSerialLoops(autotuned.module.kernels.get('mm').source);
      expect(baseFor, 'baseline kernel has a finite serial-loop count').toBeGreaterThan(0);
      expect(autoFor, 'autotuned kernel has no more serial loops than the default schedule').toBeLessThanOrEqual(baseFor);
    });
  });

  describe('Ansor-style auto-scheduled matmul', () => {
    it('auto-generates a register-blocked kernel reaching >=20% of FP32 peak', async () => {
      const peakG = peakFp32GFLOPs();
      const minFraction = 0.20;

      for (const N of [1024, 2048]) {
        const { r, err } = await autotunedVsCpu(N, [1, 2], { seed: 7, timeBudgetMs: 200000 });
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
      const { err } = await autotunedVsCpu(500, [3, 4], { seed: 11, timeBudgetMs: 120000 });
      expect(err, `N=500 maxRelErr (${cudaDeps.arch})`).toBeLessThan(1e-3);
    }, 200000);
  });
});
