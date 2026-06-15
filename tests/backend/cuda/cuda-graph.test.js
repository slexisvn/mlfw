import { describe, it, expect } from 'vitest';
import { tensor, matmul, relu, tanh } from '../../../src/index.js';
import { compile } from '../../../src/tracing/compile.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

let cudaRuntime = null;
if (cudaDeps) cudaRuntime = await import('#io/cuda_runtime');

const flat = (v) => Array.from(v && v.contiguous ? v.contiguous().data : v.data);
const maxRelErr = (a, b) => { let e = 0; for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i]))); return e; };

function mk(n, seed) { return Array.from({ length: n }, (_, i) => Math.sin(seed + i * 0.21) * 0.4); }

describe.skipIf(!cudaDeps)('CUDA graph capture/replay on the compiled plan', () => {
  it('graphed replay matches non-graphed and CPU on a native matmul chain', async () => {
    const x = tensor(mk(6 * 8, 1), { shape: [6, 8] });
    const w1 = tensor(mk(8 * 12, 2), { shape: [8, 12] });
    const w2 = tensor(mk(12 * 5, 3), { shape: [12, 5] });
    const fwd = (a, p, q) => tanh(matmul(relu(matmul(a, p)), q));

    const cpu = flat(await compile({ forward: fwd }, [x, w1, w2], { target: CPUTarget() })(x, w1, w2));

    const cf = compile({ forward: fwd }, [x, w1, w2], { target: CUDATarget() });
    expect(cf.result().module.executionPlan, 'native matmul chain produced a multi-kernel plan').toBeTruthy();

    cudaRuntime.setCudaGraphEnabled(false);
    const noGraph = flat(await cf(x, w1, w2));
    cudaRuntime.setCudaGraphEnabled(true);
    let graphed;
    try { graphed = flat(await cf(x, w1, w2)); } finally { cudaRuntime.setCudaGraphEnabled(false); }

    expect(maxRelErr(cpu, noGraph), 'plan vs CPU').toBeLessThan(2e-3);
    expect(maxRelErr(noGraph, graphed), 'graphed vs non-graphed').toBeLessThan(1e-5);
  }, 120000);

  it('repeated graphed replays are stable', async () => {
    const x = tensor(mk(4 * 6, 7), { shape: [4, 6] });
    const w1 = tensor(mk(6 * 6, 8), { shape: [6, 6] });
    const w2 = tensor(mk(6 * 3, 9), { shape: [6, 3] });
    const fwd = (a, p, q) => matmul(relu(matmul(a, p)), q);
    const cf = compile({ forward: fwd }, [x, w1, w2], { target: CUDATarget() });
    cudaRuntime.setCudaGraphEnabled(true);
    try {
      const first = flat(await cf(x, w1, w2));
      for (let i = 0; i < 5; i++) {
        expect(maxRelErr(first, flat(await cf(x, w1, w2)))).toBeLessThan(1e-6);
      }
    } finally { cudaRuntime.setCudaGraphEnabled(false); }
  }, 120000);
});
