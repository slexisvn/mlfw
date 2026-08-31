import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/compiler/support/target.js';
import { cudaDeps } from '../../_utils/cuda.js';

const T = 60000;

function rnd(n, seed) { const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = Math.sin(i * 0.37 + seed) * 0.5; return x; }

// Shapes are kept small (GN divisible by 64 so the float4 path engages, but few MFLOPs so
// the CPU reference stays fast) — these exercise the vectorized kernel without TDR risk.
function conv(N, Cin, H, W, O, Kh, Kw, st, pad, dil) {
  const Oh = Math.floor((H + 2 * pad - dil * (Kh - 1) - 1) / st) + 1;
  const Ow = Math.floor((W + 2 * pad - dil * (Kw - 1) - 1) / st) + 1;
  return {
    N, Cin, H, W, O, Kh, Kw, Oh, Ow,
    build: () => buildFunction('cv',
      [new TensorType([N, Cin, H, W], 'f32'), new TensorType([O, Cin, Kh, Kw], 'f32')],
      [new TensorType([N, O, Oh, Ow], 'f32')],
      (b, a) => b.returnOp([b.conv(a[0], a[1], [st, st], [[pad, pad], [pad, pad]], { dilation: [dil, dil] }).getResult(0)])),
  };
}

async function convErr(s, opts) {
  const input = rnd(s.N * s.Cin * s.H * s.W, 1), weight = rnd(s.O * s.Cin * s.Kh * s.Kw, 2);
  const cpu = new Float32Array(s.N * s.O * s.Oh * s.Ow), gpu = new Float32Array(s.N * s.O * s.Oh * s.Ow);
  compileGraph(s.build(), CPUTarget(), { scheduling: { enabled: true } }).run('cv', input, weight, cpu);
  await compileGraph(s.build(), CUDATarget(), opts).runAsync('cv', input, weight, gpu);
  let maxErr = 0;
  for (let i = 0; i < cpu.length; i++) maxErr = Math.max(maxErr, Math.abs(cpu[i] - gpu[i]) / (1 + Math.abs(cpu[i])));
  return maxErr;
}

const SCHED = { scheduling: { enabled: true } };

describe.skipIf(!cudaDeps)('CUDA float4 vectorized implicit-GEMM conv', () => {
  it('emits transposed-A float4 kernel for a meaty divisible conv', () => {
    const src = compileGraph(conv(8, 64, 10, 10, 64, 3, 3, 1, 0, 1).build(), CUDATarget(), SCHED).getSource('cv');
    expect(src.includes('reinterpret_cast<float4*>')).toBe(true);
    expect(src.includes('iv_As')).toBe(true); // transposed-A shared tile of the vectorized kernel
    expect(src.includes('__align__(16)')).toBe(true);
  });

  it('divisible 3x3 (Cin=64) matches CPU', async () => expect(await convErr(conv(8, 64, 10, 10, 64, 3, 3, 1, 0, 1), SCHED)).toBeLessThan(2e-3), T);
  it('Cin=128 3x3 matches CPU', async () => expect(await convErr(conv(4, 128, 10, 10, 128, 3, 3, 1, 0, 1), SCHED)).toBeLessThan(2e-3), T);
  it('Cin=256 (M=64) 3x3 matches CPU', async () => expect(await convErr(conv(4, 256, 10, 10, 64, 3, 3, 1, 0, 1), SCHED)).toBeLessThan(2e-3), T);
  it('padded (same) 3x3 matches CPU', async () => expect(await convErr(conv(8, 64, 8, 8, 64, 3, 3, 1, 1, 1), SCHED)).toBeLessThan(2e-3), T);
  it('stride-2 with padding matches CPU', async () => expect(await convErr(conv(8, 64, 16, 16, 128, 3, 3, 2, 1, 1), SCHED)).toBeLessThan(2e-3), T);
  it('dilation-2 matches CPU', async () => expect(await convErr(conv(8, 64, 12, 12, 64, 3, 3, 1, 0, 2), SCHED)).toBeLessThan(2e-3), T);
  it('1x1 (Cin=256, M=64) matches CPU', async () => expect(await convErr(conv(8, 256, 8, 8, 64, 1, 1, 1, 0, 1), SCHED)).toBeLessThan(2e-3), T);
  it('rectangular H!=W matches CPU', async () => expect(await convErr(conv(8, 64, 10, 8, 128, 3, 3, 1, 0, 1), SCHED)).toBeLessThan(2e-3), T);

  it('falls back to scalar implicit-GEMM when not vectorizable (Cin not multiple of 8)', async () => {
    const s = conv(8, 20, 12, 12, 128, 3, 3, 1, 0, 1); // K=180>=128 but Cin%8!=0 -> not vectorizable
    const src = compileGraph(s.build(), CUDATarget(), SCHED).getSource('cv');
    expect(src.includes('reinterpret_cast<float4*>')).toBe(false);
    expect(src.includes('ig_As')).toBe(true); // scalar register-block path
    expect(await convErr(s, SCHED)).toBeLessThan(2e-3);
  }, T);

  it('convNoVec forces the scalar path and stays correct', async () => {
    const NOVEC = { scheduling: { enabled: true, convNoVec: true } };
    const src = compileGraph(conv(8, 64, 10, 10, 64, 3, 3, 1, 0, 1).build(), CUDATarget(), NOVEC).getSource('cv');
    expect(src.includes('reinterpret_cast<float4*>')).toBe(false);
    expect(await convErr(conv(8, 64, 10, 10, 64, 3, 3, 1, 0, 1), NOVEC)).toBeLessThan(2e-3);
  }, T);
});
