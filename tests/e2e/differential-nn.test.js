import { describe, it, expect } from 'vitest';
import { tensor, add, mul, matmul, relu, sum, max, min } from '../../src/index.js';
import { layer_norm } from '../../src/nn/functional/normalization.js';
import { conv2d } from '../../src/nn/functional/conv.js';
import { max_pool2d, avg_pool2d } from '../../src/nn/functional/pooling.js';
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
  const sub = numel(shape.slice(1)); const out = [];
  for (let i = 0; i < shape[0]; i++) out.push(nest(flat.slice(i * sub, (i + 1) * sub), shape.slice(1)));
  return out;
}
function mk(rng, shape, lo, hi, dtype) {
  const n = numel(shape); const flat = [];
  for (let i = 0; i < n; i++) {
    let v = lo + (hi - lo) * rng();
    if (dtype && dtype.startsWith('i')) v = Math.round(v);
    flat.push(v);
  }
  return tensor(nest(flat, shape), dtype ? { dtype } : undefined);
}
function flatten(t) {
  if (t && typeof t.contiguous === 'function') return Array.from(t.contiguous().data);
  if (t && t.data) return Array.from(t.data);
  return [t];
}

async function checkDiff(prog, makeTarget, tol = 2e-3) {
  const rng = mulberry32(3000 + prog.name.length * 17);
  const inputs = prog.shapes.map((s) => mk(rng, s, prog.lo ?? -1, prog.hi ?? 1, prog.dtype));
  const eager = flatten(prog.fwd(...inputs));
  const compiled = compile({ forward: (...a) => prog.fwd(...a) }, inputs, { target: makeTarget() });
  const out = flatten(await compiled(...inputs));
  expect(out.length).toBe(eager.length);
  for (let i = 0; i < eager.length; i++) {
    const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
    expect(relErr, `${prog.name} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(tol);
  }
}

const BOTH = [
  { name: 'conv2d_k3', shapes: [[1, 2, 8, 8], [3, 2, 3, 3]], fwd: (x, w) => conv2d(x, w, null, [1, 1], [[0, 0], [0, 0]]) },
  { name: 'conv2d_pad', shapes: [[1, 2, 7, 7], [4, 2, 3, 3]], fwd: (x, w) => conv2d(x, w, null, [1, 1], [[1, 1], [1, 1]]) },
  { name: 'conv2d_stride', shapes: [[1, 1, 8, 8], [2, 1, 2, 2]], fwd: (x, w) => conv2d(x, w, null, [2, 2], [[0, 0], [0, 0]]) },
  { name: 'conv2d_bias', shapes: [[1, 2, 6, 6], [3, 2, 3, 3], [3]], fwd: (x, w, b) => conv2d(x, w, b, [1, 1], [[0, 0], [0, 0]]) },
  { name: 'maxpool', shapes: [[1, 2, 8, 8]], fwd: (x) => max_pool2d(x, [2, 2], [2, 2]) },
  { name: 'maxpool_pad', shapes: [[1, 1, 7, 7]], fwd: (x) => max_pool2d(x, [2, 2], [2, 2], [[1, 1], [1, 1]]) },
  { name: 'layernorm', shapes: [[4, 16]], fwd: (x) => layer_norm(x, [16], null, null, 1e-5) },
  { name: 'conv_relu_pool', shapes: [[1, 1, 8, 8], [2, 1, 3, 3]], fwd: (x, w) => max_pool2d(relu(conv2d(x, w, null, [1, 1], [[0, 0], [0, 0]])), [2, 2], [2, 2]) },
];

const BOTH2 = [
  { name: 'avgpool', shapes: [[1, 2, 8, 8]], fwd: (x) => avg_pool2d(x, [2, 2], [2, 2]) },
  { name: 'add_i32', dtype: 'i32', lo: -50, hi: 50, shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'mul_i32', dtype: 'i32', lo: -20, hi: 20, shapes: [[6, 8], [6, 8]], fwd: (x, y) => mul(x, y) },
  { name: 'sum_i32', dtype: 'i32', lo: -50, hi: 50, shapes: [[6, 8]], fwd: (x) => sum(x, 1) },
  { name: 'max_i32_neg', dtype: 'i32', lo: -60, hi: -1, shapes: [[6, 8]], fwd: (x) => max(x, 1) },
  { name: 'min_i32_pos', dtype: 'i32', lo: 1, hi: 60, shapes: [[6, 8]], fwd: (x) => min(x, 1) },
  { name: 'matmul_i32', dtype: 'i32', lo: -10, hi: 10, shapes: [[5, 6], [6, 4]], fwd: (x, y) => matmul(x, y) },
  { name: 'add_f64', dtype: 'f64', shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'matmul_f64', dtype: 'f64', shapes: [[5, 6], [6, 4]], fwd: (x, y) => matmul(x, y) },
  { name: 'layernorm_affine', shapes: [[4, 16], [16], [16]], fwd: (x, w, b) => layer_norm(x, [16], w, b, 1e-5) },
];

describe('differential nn: eager vs compiled (conv/pool/norm/dtype)', () => {
  for (const prog of [...BOTH, ...BOTH2]) {
    it(`${prog.name} on cpu matches eager`, () => checkDiff(prog, CPUTarget));
    it(`${prog.name} on wasm matches eager`, () => checkDiff(prog, WasmTarget));
  }
});
