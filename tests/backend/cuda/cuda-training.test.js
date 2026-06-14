import { describe, it, expect } from 'vitest';
import { tensor, matmul, relu, tanh, sigmoid, sum } from '../../../src/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from './cuda-setup.js';

const flat = (v) => Array.from(v && v.contiguous ? v.contiguous().data : v.data);

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function data(rng, s, lo, hi) {
  const n = s.reduce((a, b) => a * b, 1);
  const flatArr = [];
  for (let i = 0; i < n; i++) flatArr.push(lo + (hi - lo) * rng());
  const nest = (fl, sh) => sh.length === 1 ? fl.slice(0, sh[0]) : Array.from({ length: sh[0] }, (_, i) => nest(fl.slice(i * fl.length / sh[0], (i + 1) * fl.length / sh[0]), sh.slice(1)));
  return nest(flatArr, s);
}

const PROGRAMS = [
  { name: 'matmul', shapes: [[4, 6], [6, 5]], fwd: (x, y) => matmul(x, y) },
  { name: 'matmul_relu', shapes: [[4, 6], [6, 5]], fwd: (x, y) => relu(matmul(x, y)) },
  { name: 'mlp_chain', shapes: [[3, 4], [4, 5], [5, 2]], fwd: (x, a, b) => matmul(relu(matmul(x, a)), b) },
  { name: 'deep_chain', shapes: [[2, 4], [4, 4], [4, 3]], fwd: (x, a, b) => tanh(matmul(relu(matmul(x, a)), b)) },
  { name: 'sigmoid_mlp', shapes: [[5, 8], [8, 4]], fwd: (x, a) => sigmoid(matmul(x, a)) },
];

async function checkBackwardCuda(prog) {
  const rng = mulberry32(7000 + prog.name.length * 31);
  const datas = prog.shapes.map((s) => data(rng, s, -1, 1));
  const eagerInputs = datas.map((d) => tensor(d, { requiresGrad: true }));
  sum(prog.fwd(...eagerInputs)).backward();
  const eagerGrads = eagerInputs.map((x) => flat(x.grad));

  const inputs = datas.map((d) => tensor(d));
  const cf = compileWithBackward({ forward: (...a) => prog.fwd(...a) }, inputs, { target: CUDATarget() });
  let out = cf(...inputs); if (out && out.then) out = await out;
  const g = ones(Array.isArray(out) ? out[0].shape : out.shape);
  let cg = cf.backward(g); if (cg && cg.then) cg = await cg;
  const compiledGrads = cg.map((t) => flat(t));

  expect(compiledGrads.length).toBe(eagerGrads.length);
  let maxErr = 0;
  for (let i = 0; i < eagerGrads.length; i++) {
    expect(compiledGrads[i].length).toBe(eagerGrads[i].length);
    for (let k = 0; k < eagerGrads[i].length; k++) {
      maxErr = Math.max(maxErr, Math.abs(eagerGrads[i][k] - compiledGrads[i][k]) / (1 + Math.abs(eagerGrads[i][k])));
    }
  }
  return maxErr;
}

describe.skipIf(!cudaDeps)('CUDA training: compiled gradients match eager autograd on real GPU', () => {
  for (const prog of PROGRAMS) {
    it(`${prog.name} backward on CUDA`, async () => {
      expect(await checkBackwardCuda(prog), `${prog.name} grad maxRelErr`).toBeLessThan(3e-3);
    }, 60000);
  }

  it('big MLP (batch 64, hidden 512, 3 layers) backward on CUDA', async () => {
    const prog = { name: 'big', shapes: [[64, 512], [512, 512], [512, 128]], fwd: (x, a, b) => relu(matmul(relu(matmul(x, a)), b)) };
    expect(await checkBackwardCuda(prog), 'big mlp grad maxRelErr').toBeLessThan(3e-3);
  }, 120000);
});
