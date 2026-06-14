import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

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

describe.skipIf(!cudaDeps)('CUDA matmulBackend: cublas', () => {
  it('cublas mode dispatches matmul and matches CPU across shapes', async () => {
    for (const [M, K, N] of [[16, 24, 16], [128, 256, 64], [7, 13, 5], [256, 256, 256]]) {
      const { e, usedCublas } = await diff(M, K, N);
      expect(usedCublas, `${M}x${K}x${N} used cublas`).toBe(true);
      expect(e, `${M}x${K}x${N} maxRelErr`).toBeLessThan(2e-3);
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
    expect(e).toBeLessThan(2e-3);
  }, 60000);
});
