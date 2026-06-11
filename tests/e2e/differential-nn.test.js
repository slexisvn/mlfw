import { describe, it, expect } from 'vitest';
import { tensor, add, mul, matmul, relu, sum, mean, max, min, prod, argmax, argmin } from '../../src/index.js';
import { layer_norm, group_norm } from '../../src/nn/functional/normalization.js';
import { conv2d } from '../../src/nn/functional/conv.js';
import { max_pool2d, avg_pool2d } from '../../src/nn/functional/pooling.js';
import { compile } from '../../src/tracing/compile.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';
import { eq, where, gt, clamp, pad, one_hot, index_select, cat, stack } from '../../src/index.js';
import { isDtypeFloat } from '../../src/backend/dtype_map.js';

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
    if (dtype && !isDtypeFloat(dtype)) v = Math.round(v);
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
  { name: 'groupnorm', shapes: [[2, 4, 3, 3], [4], [4]], fwd: (x, w, b) => group_norm(x, 2, w, b, 1e-5) },
  { name: 'groupnorm_g1', shapes: [[2, 6, 2, 2], [6], [6]], fwd: (x, w, b) => group_norm(x, 1, w, b, 1e-5) },
  { name: 'conv_relu_pool', shapes: [[1, 1, 8, 8], [2, 1, 3, 3]], fwd: (x, w) => max_pool2d(relu(conv2d(x, w, null, [1, 1], [[0, 0], [0, 0]])), [2, 2], [2, 2]) },
];

const BOTH2 = [
  { name: 'avgpool', shapes: [[1, 2, 8, 8]], fwd: (x) => avg_pool2d(x, [2, 2], [2, 2]) },
  { name: 'add_i32', dtype: 'i32', lo: -50, hi: 50, shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'mul_i32', dtype: 'i32', lo: -20, hi: 20, shapes: [[6, 8], [6, 8]], fwd: (x, y) => mul(x, y) },
  { name: 'sum_i32', dtype: 'i32', lo: -50, hi: 50, shapes: [[6, 8]], fwd: (x) => sum(x, 1) },
  { name: 'max_i32_neg', dtype: 'i32', lo: -60, hi: -1, shapes: [[6, 8]], fwd: (x) => max(x, 1) },
  { name: 'min_i32_pos', dtype: 'i32', lo: 1, hi: 60, shapes: [[6, 8]], fwd: (x) => min(x, 1) },
  { name: 'prod_f64', dtype: 'f64', lo: -1.5, hi: 1.5, shapes: [[4, 5]], fwd: (x) => prod(x, 1) },
  { name: 'prod_f32', lo: -1.5, hi: 1.5, shapes: [[4, 5]], fwd: (x) => prod(x, 1) },
  { name: 'argmax_dim1', shapes: [[5, 7]], fwd: (x) => argmax(x, 1) },
  { name: 'argmin_dim0', shapes: [[5, 7]], fwd: (x) => argmin(x, 0) },
  { name: 'argmax_keepdim', shapes: [[4, 6]], fwd: (x) => argmax(x, 1, true) },
  { name: 'mean_negdim', shapes: [[2, 3, 4]], fwd: (x) => mean(x, -1) },
  { name: 'sum_negdim2', shapes: [[2, 3, 4]], fwd: (x) => sum(x, -2) },
  { name: 'prod_negdim_f64', dtype: 'f64', lo: -1.2, hi: 1.2, shapes: [[3, 4]], fwd: (x) => prod(x, -1) },
  { name: 'max_negdim_keepdim', shapes: [[2, 3, 4]], fwd: (x) => max(x, -1, true) },
  { name: 'matmul_i32', dtype: 'i32', lo: -10, hi: 10, shapes: [[5, 6], [6, 4]], fwd: (x, y) => matmul(x, y) },
  { name: 'add_i16', dtype: 'i16', lo: -20000, hi: 20000, shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'mul_i16', dtype: 'i16', lo: -150, hi: 150, shapes: [[6, 8], [6, 8]], fwd: (x, y) => mul(x, y) },
  { name: 'sum_i16', dtype: 'i16', lo: -2000, hi: 2000, shapes: [[6, 8]], fwd: (x) => sum(x, 1) },
  { name: 'max_i16', dtype: 'i16', lo: -30000, hi: 30000, shapes: [[6, 8]], fwd: (x) => max(x, 1) },
  { name: 'matmul_i16', dtype: 'i16', lo: -20, hi: 20, shapes: [[5, 6], [6, 4]], fwd: (x, y) => matmul(x, y) },
  { name: 'add_ui8', dtype: 'ui8', lo: 0, hi: 120, shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'mul_ui8', dtype: 'ui8', lo: 0, hi: 15, shapes: [[6, 8], [6, 8]], fwd: (x, y) => mul(x, y) },
  { name: 'sum_ui8', dtype: 'ui8', lo: 0, hi: 30, shapes: [[6, 8]], fwd: (x) => sum(x, 1) },
  { name: 'max_ui8', dtype: 'ui8', lo: 0, hi: 255, shapes: [[6, 8]], fwd: (x) => max(x, 1) },
  { name: 'matmul_ui8', dtype: 'ui8', lo: 0, hi: 6, shapes: [[5, 6], [6, 4]], fwd: (x, y) => matmul(x, y) },
  { name: 'eq_bool', dtype: 'bool', lo: 0, hi: 1, shapes: [[6, 8], [6, 8]], fwd: (x, y) => eq(x, y) },
  { name: 'add_f64', dtype: 'f64', shapes: [[6, 8], [6, 8]], fwd: (x, y) => add(x, y) },
  { name: 'matmul_f64', dtype: 'f64', shapes: [[5, 6], [6, 4]], fwd: (x, y) => matmul(x, y) },
  { name: 'layernorm_affine', shapes: [[4, 16], [16], [16]], fwd: (x, w, b) => layer_norm(x, [16], w, b, 1e-5) },
  { name: 'clamp_scalar', lo: -2, hi: 2, shapes: [[4, 6]], fwd: (x) => clamp(x, -0.5, 0.5) },
  { name: 'pad_2d', shapes: [[3, 4]], fwd: (x) => pad(x, [1, 2], [2, 1], 0) },
  { name: 'repeat_2d', shapes: [[2, 3]], fwd: (x) => x.repeat(2, 3) },
  { name: 'repeat_1d', shapes: [[4]], fwd: (x) => x.repeat(3) },
  { name: 'tile_promote', shapes: [[3]], fwd: (x) => x.tile([2, 2]) },
];

describe('differential nn: eager vs compiled (conv/pool/norm/dtype)', () => {
  for (const prog of [...BOTH, ...BOTH2]) {
    it(`${prog.name} on cpu matches eager`, () => checkDiff(prog, CPUTarget));
    it(`${prog.name} on wasm matches eager`, () => checkDiff(prog, WasmTarget));
  }
});

describe('index ops: eager vs compiled (one_hot/index_select)', () => {
  async function checkIndexDiff(name, makeTarget, inputs, fwd, tol = 1e-6) {
    const eager = flatten(fwd(...inputs));
    const compiled = compile({ forward: (...a) => fwd(...a) }, inputs, { target: makeTarget() });
    const out = flatten(await compiled(...inputs));
    expect(out.length).toBe(eager.length);
    for (let i = 0; i < eager.length; i++) {
      const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
      expect(relErr, `${name} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(tol);
    }
  }

  for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`one_hot on ${tname} matches eager`, () => {
      const idx = tensor([0, 2, 1, 2], { dtype: 'i32' });
      return checkIndexDiff('one_hot', T, [idx], (a) => one_hot(a, 3));
    });

    it(`index_select dim0 on ${tname} matches eager`, () => {
      const x = tensor([[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]]);
      const idx = tensor([3, 0, 2], { dtype: 'i32' });
      return checkIndexDiff('index_select_d0', T, [x, idx], (a, b) => index_select(a, 0, b));
    });

    it(`index_select dim1 on ${tname} matches eager`, () => {
      const x = tensor([[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]]);
      const idx = tensor([2, 0], { dtype: 'i32' });
      return checkIndexDiff('index_select_d1', T, [x, idx], (a, b) => index_select(a, 1, b));
    });

    it(`cat dim0 on ${tname} matches eager`, () => {
      const x = tensor([[1, 2], [3, 4]]);
      const y = tensor([[5, 6], [7, 8]]);
      return checkIndexDiff('cat0', T, [x, y], (a, b) => cat([a, b], 0));
    });

    it(`cat dim1 on ${tname} matches eager`, () => {
      const x = tensor([[1, 2], [3, 4]]);
      const y = tensor([[5, 6], [7, 8]]);
      return checkIndexDiff('cat1', T, [x, y], (a, b) => cat([a, b], 1));
    });

    it(`stack dim0 on ${tname} matches eager`, () => {
      const x = tensor([[1, 2], [3, 4]]);
      const y = tensor([[5, 6], [7, 8]]);
      return checkIndexDiff('stack0', T, [x, y], (a, b) => stack([a, b], 0));
    });

    it(`stack dim1 on ${tname} matches eager`, () => {
      const x = tensor([[1, 2], [3, 4]]);
      const y = tensor([[5, 6], [7, 8]]);
      return checkIndexDiff('stack1', T, [x, y], (a, b) => stack([a, b], 1));
    });

    it(`split then cat round-trips on ${tname}`, () => {
      const x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
      return checkIndexDiff('split_cat', T, [x], (a) => cat(a.split(2, 1), 1));
    });

    it(`chunk then cat round-trips on ${tname}`, () => {
      const x = tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]);
      return checkIndexDiff('chunk_cat', T, [x], (a) => cat(a.chunk(2, 1), 1));
    });
  }
});

describe('composition ops: independent correctness (not eager-vs-compiled)', () => {
  it('group_norm normalizes each group to mean~0 var~1', () => {
    const x = tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[-2, -1], [0, 1]], [[10, 20], [30, 40]]]]);
    const w = tensor([1, 1, 1, 1]);
    const b = tensor([0, 0, 0, 0]);
    const out = Array.from(group_norm(x, 2, w, b, 1e-5).contiguous().data);
    for (const g of [out.slice(0, 8), out.slice(8, 16)]) {
      const mean = g.reduce((a, c) => a + c, 0) / g.length;
      const varr = g.reduce((a, c) => a + (c - mean) * (c - mean), 0) / g.length;
      expect(Math.abs(mean)).toBeLessThan(1e-4);
      expect(Math.abs(varr - 1)).toBeLessThan(2e-3);
    }
  });

  it('group_norm affine applies per-channel weight/bias', () => {
    const x = tensor([[[[1, 2]], [[3, 4]]]]);
    const w = tensor([2, 3]);
    const b = tensor([10, 20]);
    const out = Array.from(group_norm(x, 1, w, b, 1e-5).contiguous().data);
    const base = Array.from(group_norm(x, 1, tensor([1, 1]), tensor([0, 0]), 1e-5).contiguous().data);
    expect(out[0]).toBeCloseTo(base[0] * 2 + 10, 4);
    expect(out[2]).toBeCloseTo(base[2] * 3 + 20, 4);
  });

  it('repeat tiles values in PyTorch order', () => {
    const x = tensor([[1, 2], [3, 4]]);
    expect(Array.from(x.repeat(2, 3).contiguous().data)).toEqual(
      [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]);
    expect(x.repeat(2, 3).shape).toEqual([4, 6]);
  });

  it('split/chunk produce correct pieces and shapes', () => {
    const x = tensor([[1, 2, 3, 4, 5]]);
    const parts = x.split([2, 3], 1);
    expect(parts.map((p) => Array.from(p.contiguous().data))).toEqual([[1, 2], [3, 4, 5]]);
    const c = tensor([[1, 2, 3, 4, 5]]).chunk(2, 1);
    expect(c.map((p) => p.shape[1])).toEqual([3, 2]);
  });

  it('index_select gathers rows', () => {
    const x = tensor([[0, 1], [10, 11], [20, 21]]);
    const out = index_select(x, 0, tensor([2, 0], { dtype: 'i32' }));
    expect(Array.from(out.contiguous().data)).toEqual([20, 21, 0, 1]);
  });

  it('one_hot sets exactly one position per index', () => {
    const out = Array.from(one_hot(tensor([0, 2, 1], { dtype: 'i32' }), 3).contiguous().data);
    expect(out).toEqual([1, 0, 0, 0, 0, 1, 0, 1, 0]);
  });

  it('clamp bounds values', () => {
    const out = Array.from(clamp(tensor([[-2, 0.2], [0.8, 5]]), -0.5, 0.5).contiguous().data);
    const exp = [-0.5, 0.2, 0.5, 0.5];
    for (let i = 0; i < exp.length; i++) expect(out[i]).toBeCloseTo(exp[i], 5);
  });
});

describe('eager where with bool-dtype condition uses operand dtype for output', () => {
  it('does not wrap float operands through the condition dtype', () => {
    const cond = tensor([[1, 0], [0, 1]], { dtype: 'bool' });
    const a = tensor([[10, 20], [30, 40]]);
    const b = tensor([[-1, -2], [-3, -4]]);
    const out = where(cond, a, b);
    expect(out.dtype).toBe('f32');
    expect(Array.from(out.contiguous().data)).toEqual([10, -2, -3, 40]);
  });
});
