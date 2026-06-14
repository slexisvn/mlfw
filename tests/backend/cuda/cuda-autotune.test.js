import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

const F32 = ScalarType.F32;
const rnd = (n, s) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.01 + s));

describe.skipIf(!cudaDeps)('CUDA hardware-measured autotune', () => {
  it('autotune with hardwareMeasure compiles, completes, and matches CPU', async () => {
    const N = 256;
    const t = new TensorType([N, N], F32);
    const mk = () => buildFunction('mm', [t, t], [t], (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]));
    const a = rnd(N * N, 1), b = rnd(N * N, 2);
    const cpu = new Float32Array(N * N), gpu = new Float32Array(N * N);
    compileGraph(mk(), CPUTarget(), { scheduling: { enabled: true } }).run('mm', a, b, cpu);
    const r = compileGraph(mk(), CUDATarget(), { scheduling: { enabled: true, autotune: true, seed: 7, hardwareMeasure: true } });
    await r.runAsync('mm', a, b, gpu);
    let maxErr = 0;
    for (let i = 0; i < cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
    expect(maxErr, `autotune-measured matmul maxRelErr (${cudaDeps.arch})`).toBeLessThan(2e-3);
  }, 60000);
});
