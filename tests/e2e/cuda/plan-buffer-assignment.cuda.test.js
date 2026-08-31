import { describe, it, expect } from 'vitest';
import { tensor, matmul, relu, tanh } from '../../../src/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compile } from '../../../src/tracing/compile.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CUDATarget } from '../../../src/compiler/support/target.js';
import { liveBytes, peakLiveBytes, resetPeakLiveBytes, drainPool } from '../../../src/runtime/node/cuda/memory.js';
import { cudaDeps } from '../../_utils/cuda.js';
import { mulberry32 } from '../../_utils/rng.js';

const flat = (v) => Array.from(v && v.contiguous ? v.contiguous().data : v.data);
const maxAbsErr = (a, b) => {
  let e = 0;
  for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]));
  return e;
};

function data(rng, shape) {
  const n = shape.reduce((a, b) => a * b, 1);
  const flatArr = [];
  for (let i = 0; i < n; i++) flatArr.push(-1 + 2 * rng());
  const nest = (fl, sh) => sh.length === 1 ? fl.slice(0, sh[0])
    : Array.from({ length: sh[0] }, (_, i) => nest(fl.slice(i * fl.length / sh[0], (i + 1) * fl.length / sh[0]), sh.slice(1)));
  return nest(flatArr, shape);
}

describe.skipIf(!cudaDeps)('CUDA module-level plan buffer assignment', () => {
  const B = 64, W = 192, OUT = 96;
  const rng = mulberry32(20260812);
  const inputs = [[B, W], [W, W], [W, W], [W, OUT]].map((s) => tensor(data(rng, s)));
  const fwd = (x, a, b, c) => tanh(matmul(relu(matmul(relu(matmul(x, a)), b)), c));

  async function trainStep(opts) {
    drainPool();
    resetPeakLiveBytes();
    const base = liveBytes();

    const cf = compileWithBackward({ forward: fwd }, inputs, { target: CUDATarget(), ...opts });
    let out = cf(...inputs);
    if (out && out.then) out = await out;
    let grads = cf.backward(ones(out.shape));
    if (grads && grads.then) grads = await grads;

    return { peak: peakLiveBytes() - base, out: flat(out), grads: grads.map(flat) };
  }

  async function forwardPlan(opts) {
    const cf = compile({ forward: fwd }, inputs, { target: CUDATarget(), ...opts });
    const out = flat(await cf(...inputs));
    return { plan: cf.result().module.executionPlan, out };
  }

  it('a training step peaks lower with slot reuse and stays bit-identical', async () => {
    const off = await trainStep({ memory: { planReuse: false } });
    const on = await trainStep({});

    expect(maxAbsErr(on.out, off.out), 'forward is bit-identical').toBe(0);
    expect(on.grads.length).toBe(off.grads.length);
    for (let i = 0; i < off.grads.length; i++) {
      expect(on.grads[i].length).toBe(off.grads[i].length);
      expect(maxAbsErr(on.grads[i], off.grads[i]), `grad ${i} is bit-identical`).toBe(0);
    }

    expect(off.peak, 'training step allocates device memory').toBeGreaterThan(0);
    expect(on.peak, `peak device bytes ${on.peak} vs ${off.peak} (${cudaDeps.arch})`).toBeLessThan(off.peak);
  }, 300000);

  it('attaches the assignment to the compiled plan and donates elementwise steps', async () => {
    const off = await forwardPlan({ memory: { planReuse: false } });
    expect(off.plan, 'graph split into a plan').toBeTruthy();
    expect(off.plan.buffers, 'reuse disabled').toBeUndefined();

    const noDonation = await forwardPlan({ memory: { planDonation: false } });
    expect(noDonation.plan.buffers, 'reuse enabled').toBeTruthy();
    expect(noDonation.plan.buffers.bufferBytes.length).toBeLessThan(noDonation.plan.numSlots);
    expect(noDonation.plan.buffers.donated).toBe(0);

    const on = await forwardPlan({});
    expect(on.plan.buffers.donated, 'elementwise steps donate').toBeGreaterThan(0);
    expect(on.plan.buffers.bufferBytes.length).toBeLessThanOrEqual(noDonation.plan.buffers.bufferBytes.length);

    expect(maxAbsErr(noDonation.out, off.out), 'reuse is bit-identical').toBe(0);
    expect(maxAbsErr(on.out, off.out), 'donation is bit-identical').toBe(0);
  }, 300000);
});
