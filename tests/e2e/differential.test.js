import { describe, it, expect } from 'vitest';
import {
  tensor, add, sub, mul, div, neg, exp, log, sqrt, abs, tanh, sigmoid,
  relu, gelu, silu, sum, mean, max, min, prod, matmul, softmax, log_softmax,
} from '../../src/index.js';
import { compile } from '../../src/tracing/compile.js';
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
  const subN = numel(shape.slice(1));
  const out = [];
  for (let i = 0; i < shape[0]; i++) out.push(nest(flat.slice(i * subN, (i + 1) * subN), shape.slice(1)));
  return out;
}
function mk(rng, shape, lo, hi) {
  const n = numel(shape);
  const flat = [];
  for (let i = 0; i < n; i++) flat.push(lo + (hi - lo) * rng());
  return tensor(nest(flat, shape));
}
function flatten(v) {
  if (v && typeof v.contiguous === 'function') return Array.from(v.contiguous().data);
  if (v && v.data) return Array.from(v.data);
  if (Array.isArray(v)) return v.flat(Infinity);
  return [v];
}

const PROGRAMS = [
  { name: 'add', shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'mul_sub', shapes: [[6, 8], [6, 8]], fwd: (x, y) => sub(mul(x, y), x) },
  { name: 'div', shapes: [[5, 5], [5, 5]], lo: 1, hi: 2, fwd: (x, y) => div(x, y) },
  { name: 'relu', shapes: [[7, 9]], fwd: (x) => relu(x) },
  { name: 'sigmoid', shapes: [[7, 9]], fwd: (x) => sigmoid(x) },
  { name: 'tanh', shapes: [[7, 9]], fwd: (x) => tanh(x) },
  { name: 'gelu', shapes: [[7, 9]], fwd: (x) => gelu(x) },
  { name: 'silu', shapes: [[7, 9]], fwd: (x) => silu(x) },
  { name: 'exp', shapes: [[7, 9]], fwd: (x) => exp(x) },
  { name: 'log', shapes: [[7, 9]], lo: 0.5, hi: 3, fwd: (x) => log(x) },
  { name: 'sqrt', shapes: [[7, 9]], lo: 0.5, hi: 3, fwd: (x) => sqrt(x) },
  { name: 'abs_neg', shapes: [[7, 9]], fwd: (x) => abs(neg(x)) },
  { name: 'matmul', shapes: [[6, 10], [10, 7]], fwd: (x, y) => matmul(x, y) },
  { name: 'matmul_relu', shapes: [[8, 12], [12, 5]], fwd: (x, y) => relu(matmul(x, y)) },
  { name: 'matmul_chain', shapes: [[4, 6], [6, 8], [8, 3]], fwd: (x, y, z) => matmul(matmul(x, y), z) },
  { name: 'softmax', shapes: [[6, 11]], fwd: (x) => softmax(x, 1) },
  { name: 'log_softmax', shapes: [[6, 11]], fwd: (x) => log_softmax(x, 1) },
  { name: 'sum_axis', shapes: [[6, 8]], fwd: (x) => sum(x, 1) },
  { name: 'mean_axis', shapes: [[6, 8]], fwd: (x) => mean(x, 1) },
  { name: 'max_axis', shapes: [[6, 8]], fwd: (x) => max(x, 1) },
  { name: 'min_axis', shapes: [[6, 8]], fwd: (x) => min(x, 1) },
  { name: 'prod_axis', shapes: [[5, 4]], lo: 0.8, hi: 1.2, fwd: (x) => prod(x, 1) },
  { name: 'sum_full', shapes: [[6, 8]], fwd: (x) => sum(x) },
  { name: 'mean_full', shapes: [[6, 8]], fwd: (x) => mean(x) },
  { name: 'broadcast_add', shapes: [[6, 8], [1, 8]], fwd: (x, y) => add(x, y) },
  { name: 'transpose', shapes: [[5, 7]], fwd: (x) => x.transpose(0, 1) },
  { name: 'transpose_matmul', shapes: [[6, 5], [6, 7]], fwd: (x, y) => matmul(x.transpose(0, 1), y) },
  { name: 'matmul_transpose_rhs', shapes: [[4, 6], [5, 6]], fwd: (x, y) => matmul(x, y.transpose(0, 1)) },
  { name: 'permute', shapes: [[2, 3, 4]], fwd: (x) => x.permute(2, 0, 1) },
  { name: 'reshape', shapes: [[4, 6]], fwd: (x) => x.reshape([3, 8]) },
  { name: 'reshape_infer', shapes: [[4, 6]], fwd: (x) => x.reshape([-1, 4]) },
  { name: 'unsqueeze', shapes: [[4, 6]], fwd: (x) => x.unsqueeze(1) },
  { name: 'squeeze', shapes: [[4, 6]], fwd: (x) => x.reshape([4, 1, 6]).squeeze(1) },
  { name: 'slice', shapes: [[4, 8]], fwd: (x) => x.slice(1, 2, 6, 1) },
  { name: 'expand', shapes: [[1, 6]], fwd: (x) => x.expand([5, 6]) },
  { name: 'reshape_relu_matmul', shapes: [[2, 12], [6, 5]], fwd: (x, y) => relu(matmul(x.reshape([4, 6]), y)) },
];

const TARGETS = { cpu: CPUTarget, wasm: WasmTarget };

describe('differential: eager vs compiled (CPU + WASM)', () => {
  for (const prog of PROGRAMS) {
    for (const [tname, makeTarget] of Object.entries(TARGETS)) {
      it(`${prog.name} on ${tname} matches eager`, async () => {
        const rng = mulberry32(1000 + prog.name.length * 13);
        const inputs = prog.shapes.map((s) => mk(rng, s, prog.lo ?? -1, prog.hi ?? 1));
        const eager = flatten(prog.fwd(...inputs));

        const model = { forward: (...args) => prog.fwd(...args) };
        const compiled = compile(model, inputs, { target: makeTarget() });
        const out = flatten(await compiled(...inputs));

        expect(out.length).toBe(eager.length);
        for (let i = 0; i < eager.length; i++) {
          const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
          expect(relErr, `${prog.name}/${tname} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(1e-3);
        }
      });
    }
  }
});
