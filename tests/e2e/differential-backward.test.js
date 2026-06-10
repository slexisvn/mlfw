import { describe, it, expect } from 'vitest';
import { tensor, add, sub, mul, div, neg, sqrt, exp, tanh, sigmoid, relu, gelu, silu, matmul, sum } from '../../src/index.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../src/tracing/compile_backward.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function numel(s) { return s.reduce((a, b) => a * b, 1); }
function nest(flat, shape) {
  if (shape.length === 1) return flat.slice(0, shape[0]);
  const sub = numel(shape.slice(1)); const out = [];
  for (let i = 0; i < shape[0]; i++) out.push(nest(flat.slice(i * sub, (i + 1) * sub), shape.slice(1)));
  return out;
}
function data(rng, shape, lo, hi) {
  const n = numel(shape); const f = [];
  for (let i = 0; i < n; i++) f.push(lo + (hi - lo) * rng());
  return nest(f, shape);
}
function flat(t) {
  if (t && typeof t.contiguous === 'function') return Array.from(t.contiguous().data);
  return Array.from(t.data);
}

const PROGRAMS = [
  { name: 'add', shapes: [[4, 5], [4, 5]], fwd: (x, y) => add(x, y) },
  { name: 'mul_sub', shapes: [[4, 5], [4, 5]], fwd: (x, y) => sub(mul(x, y), x) },
  { name: 'div', shapes: [[4, 5], [4, 5]], lo: 0.5, hi: 2, fwd: (x, y) => div(x, y) },
  { name: 'relu', shapes: [[4, 5]], fwd: (x) => relu(x) },
  { name: 'sigmoid', shapes: [[4, 5]], fwd: (x) => sigmoid(x) },
  { name: 'tanh', shapes: [[4, 5]], fwd: (x) => tanh(x) },
  { name: 'exp', shapes: [[4, 5]], fwd: (x) => exp(x) },
  { name: 'gelu', shapes: [[4, 5]], fwd: (x) => gelu(x) },
  { name: 'silu', shapes: [[4, 5]], fwd: (x) => silu(x) },
  { name: 'sqrt', shapes: [[4, 5]], lo: 0.5, hi: 2, fwd: (x) => sqrt(x) },
  { name: 'neg_sq', shapes: [[4, 5]], fwd: (x) => neg(mul(x, x)) },
  { name: 'matmul', shapes: [[4, 6], [6, 5]], fwd: (x, y) => matmul(x, y) },
  { name: 'matmul_relu', shapes: [[4, 6], [6, 5]], fwd: (x, y) => relu(matmul(x, y)) },
  { name: 'matmul_sigmoid', shapes: [[4, 6], [6, 5]], fwd: (x, y) => sigmoid(matmul(x, y)) },
  { name: 'mlp_chain', shapes: [[3, 4], [4, 5], [5, 2]], fwd: (x, a, b) => matmul(relu(matmul(x, a)), b) },
  { name: 'deep_chain', shapes: [[2, 4], [4, 4], [4, 3]], fwd: (x, a, b) => tanh(matmul(relu(matmul(x, a)), b)) },
];

async function checkBackward(prog, makeTarget) {
  const rng = mulberry32(9000 + prog.name.length * 31);
  const datas = prog.shapes.map((s) => data(rng, s, prog.lo ?? -1, prog.hi ?? 1));

  const eagerInputs = datas.map((d) => tensor(d, { requiresGrad: true }));
  const eagerOut = prog.fwd(...eagerInputs);
  sum(eagerOut).backward();
  const eagerGrads = eagerInputs.map((x) => flat(x.grad));

  const inputs = datas.map((d) => tensor(d));
  const cf = compileWithBackward({ forward: (...a) => prog.fwd(...a) }, inputs, { target: makeTarget() });
  const out = cf(...inputs);
  const g = ones(Array.isArray(out) ? out[0].shape : out.shape);
  const compiledGrads = cf.backward(g).map((t) => flat(t));

  expect(compiledGrads.length).toBe(eagerGrads.length);
  for (let i = 0; i < eagerGrads.length; i++) {
    expect(compiledGrads[i].length).toBe(eagerGrads[i].length);
    for (let k = 0; k < eagerGrads[i].length; k++) {
      const relErr = Math.abs(eagerGrads[i][k] - compiledGrads[i][k]) / (1 + Math.abs(eagerGrads[i][k]));
      expect(relErr, `${prog.name} grad input ${i} idx ${k}: eager=${eagerGrads[i][k]} compiled=${compiledGrads[i][k]}`).toBeLessThan(3e-3);
    }
  }
}

describe('differential backward: compiled gradients vs eager autograd', () => {
  for (const prog of PROGRAMS) {
    it(`${prog.name} backward on cpu matches eager`, () => checkBackward(prog, CPUTarget));
    it(`${prog.name} backward on wasm matches eager`, () => checkBackward(prog, WasmTarget));
  }
});
