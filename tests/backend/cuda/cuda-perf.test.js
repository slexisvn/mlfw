import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

const F32 = ScalarType.F32;

function matmul(N) {
  return buildFunction('mm', [new TensorType([N, N], F32), new TensorType([N, N], F32)], [new TensorType([N, N], F32)], (b, a) => b.returnOp([b.relu(b.matmul(a[0], a[1]).getResult(0)).getResult(0)]));
}

async function timeConfig(N, opts, reps) {
  const a = new Float32Array(N * N).map((_, i) => Math.sin(i * 0.01));
  const b = new Float32Array(N * N).map((_, i) => Math.cos(i * 0.01));
  const out = new Float32Array(N * N);
  const r = compileGraph(matmul(N), CUDATarget(), opts);
  await r.runAsync('mm', a, b, out);
  await r.runAsync('mm', a, b, out);
  const samples = [];
  for (let s = 0; s < 5; s++) {
    const t0 = performance.now();
    for (let i = 0; i < reps; i++) await r.runAsync('mm', a, b, out);
    samples.push((performance.now() - t0) / reps);
  }
  samples.sort((x, y) => x - y);
  return samples[2];
}

describe.skipIf(!cudaDeps)('CUDA performance: scheduling speeds up matmul on real GPU', () => {
  it('scheduled matmul is much faster than unscheduled baseline (1-thread)', async () => {
    const N = 128;
    const base = await timeConfig(N, {}, 5);
    const sched = await timeConfig(N, { scheduling: { enabled: true } }, 20);
    const autotune = await timeConfig(N, { scheduling: { enabled: true, autotune: true, seed: 7 } }, 20);
    process.stdout.write(`\n  matmul+relu ${N}x${N} on ${cudaDeps.arch}: baseline=${base.toFixed(3)}ms sched=${sched.toFixed(3)}ms (${(base / sched).toFixed(1)}x) autotune=${autotune.toFixed(3)}ms (${(base / autotune).toFixed(1)}x vs base, ${(sched / autotune).toFixed(2)}x vs sched)\n`);
    expect(sched).toBeLessThan(base);
    expect(autotune).toBeLessThan(base);
  }, 60000);
});
