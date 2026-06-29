import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

function mk(M, K, N, relu = false) {
  return buildFunction('mm', [new TensorType([M, K], 'f32'), new TensorType([K, N], 'f32')], [new TensorType([M, N], 'f32')],
    (b, a) => { let c = b.matmul(a[0], a[1]).getResult(0); if (relu) c = b.relu(c).getResult(0); return b.returnOp([c]); });
}
function rnd(n, seed) { const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = Math.sin(i * 0.3 + seed) * 0.5; return x; }

async function matchesCpu(M, K, N, opts, relu = false, tol = 2e-3) {
  const A = rnd(M * K, 1), B = rnd(K * N, 2);
  const cpu = new Float32Array(M * N), gpu = new Float32Array(M * N);
  compileGraph(mk(M, K, N, relu), CPUTarget(), opts).run('mm', A, B, cpu);
  await compileGraph(mk(M, K, N, relu), CUDATarget(), opts).runAsync('mm', A, B, gpu);
  let maxErr = 0;
  for (let i = 0; i < cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
  return maxErr;
}

describe.skipIf(!cudaDeps)('CUDA native matmul shared-memory tiling', () => {
  const SCHED = { scheduling: { enabled: true } };
  const PRIM = { scheduling: { enabled: true, primitiveMatmul: true } };

  describe('Phase A — deterministic register-blocked (default scheduled native)', () => {
    it('square 256 matches CPU', async () => expect(await matchesCpu(256, 256, 256, SCHED)).toBeLessThan(2e-3));
    it('rectangular non-divisible 250x130x200 matches CPU', async () => expect(await matchesCpu(250, 130, 200, SCHED)).toBeLessThan(2e-3));
    it('small 64 matches CPU', async () => expect(await matchesCpu(64, 64, 64, SCHED)).toBeLessThan(2e-3));
    it('relu epilogue falls back yet stays correct', async () => expect(await matchesCpu(256, 256, 256, SCHED, true)).toBeLessThan(2e-3));
    it('emits __shared__ and __syncthreads for native scheduled matmul', () => {
      const src = compileGraph(mk(256, 256, 256), CUDATarget(), SCHED).getSource('mm');
      expect(src.includes('__shared__')).toBe(true);
      expect(src.includes('__syncthreads')).toBe(true);
    });
    it('does NOT trigger register-block path for plain elementwise', () => {
      const ew = buildFunction('ew', [new TensorType([1024], 'f32')], [new TensorType([1024], 'f32')], (b, a) => b.returnOp([b.relu(a[0]).getResult(0)]));
      expect(compileGraph(ew, CUDATarget(), SCHED).getSource('ew').includes('rb_As')).toBe(false);
    });
  });

  describe('Phase B — primitive tiled-shared (primitiveMatmul flag)', () => {
    it('divisible square 256 matches CPU', async () => expect(await matchesCpu(256, 256, 256, PRIM)).toBeLessThan(2e-3));
    it('divisible rectangular 512x256x128 matches CPU', async () => expect(await matchesCpu(512, 256, 128, PRIM)).toBeLessThan(2e-3));
    it('non-divisible falls back to Phase A and stays correct', async () => expect(await matchesCpu(250, 130, 200, PRIM)).toBeLessThan(2e-3));
    it('emits tiled-shared kernel (ts_As)', () => {
      const src = compileGraph(mk(256, 256, 256), CUDATarget(), PRIM).getSource('mm');
      expect(src.includes('ts_As')).toBe(true);
      expect(src.includes('__syncthreads')).toBe(true);
    });
  });

  describe('cuBLAS path is unaffected', () => {
    it('cublas backend does not emit a native register-block / tiled kernel', () => {
      const src = compileGraph(mk(256, 256, 256), CUDATarget(), { matmulBackend: 'cublas' }).getSource('mm');
      expect(src.includes('rb_As')).toBe(false);
      expect(src.includes('ts_As')).toBe(false);
    });
  });
});
