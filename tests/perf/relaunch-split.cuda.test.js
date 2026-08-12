import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { compile } from '../../src/tracing/compile.js';
import { CUDATarget } from '../../src/backend/target.js';
import { cudaDeps } from '../_utils/cuda.js';

const CHANNELS = 32;
const SPATIAL = 32;
const ITERS = 5;
const MIN_SPEEDUP = 5;

function rng(s) { let x = s >>> 0; return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; }; }
function data(r, sh) {
  const n = sh.reduce((a, b) => a * b, 1); const f = new Array(n);
  for (let i = 0; i < n; i++) f[i] = r() * 2 - 1;
  const nest = (fl, s) => s.length === 1 ? fl.slice(0, s[0]) : Array.from({ length: s[0] }, (_, i) => nest(fl.slice(i * fl.length / s[0], (i + 1) * fl.length / s[0]), s.slice(1)));
  return nest(f, sh);
}
const flat = (v) => Array.from(v && typeof v.contiguous === 'function' ? v.contiguous().data : v.data);

async function measure(fwd, x, splitSerializedKernels) {
  const compiled = compile({ forward: fwd }, [x], {
    target: CUDATarget(),
    scheduling: { enabled: true },
    optimization: { splitSerializedKernels },
  });
  const module = compiled.result().module;
  const threads = module.listKernels().map((n) => {
    const md = module.getKernelMetadata(n);
    return md.blockDim.reduce((a, b) => a * b, 1) * md.gridDim.reduce((a, b) => a * b, 1);
  });

  let out = compiled(x); if (out && out.then) out = await out;
  const t0 = performance.now();
  for (let i = 0; i < ITERS; i++) { let r = compiled(x); if (r && r.then) r = await r; }
  return { ms: (performance.now() - t0) / ITERS, threads, output: flat(out) };
}

describe.skipIf(!cudaDeps)('splitting a kernel that cannot run at its launch geometry beats serializing it', () => {
  it('groupnorm->conv is faster split, and bit-identical', async () => {
    const gn = new nn.GroupNorm(4, CHANNELS); gn.eval();
    const cv = new nn.Conv2d(CHANNELS, CHANNELS, 3, 1, 1); cv.eval();
    const fwd = (a) => cv.forward(gn.forward(a));
    const x = tensor(data(rng(1), [1, CHANNELS, SPATIAL, SPATIAL]));

    const serialized = await measure(fwd, x, false);
    const split = await measure(fwd, x, true);

    expect(Math.max(...serialized.threads), 'baseline was expected to be a single-thread kernel').toBe(1);
    expect(Math.max(...split.threads)).toBeGreaterThan(1024);

    let maxErr = 0;
    for (let i = 0; i < serialized.output.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(serialized.output[i] - split.output[i]));
    }
    expect(maxErr, 'splitting must not change numerics').toBe(0);

    const speedup = serialized.ms / split.ms;
    console.log(`groupnorm->conv C=${CHANNELS} S=${SPATIAL}: serialized ${serialized.ms.toFixed(2)}ms (1 kernel) -> split ${split.ms.toFixed(2)}ms (${split.threads.length} kernels) = ${speedup.toFixed(1)}x`);
    expect(speedup, `split kernels regressed to ${speedup.toFixed(1)}x`).toBeGreaterThan(MIN_SPEEDUP);
  }, 300000);
});
