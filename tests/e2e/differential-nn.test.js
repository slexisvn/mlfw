import { describe, it, expect } from 'vitest';
import { tensor, add, sub, mul, div, matmul, relu, sum, mean, max, min, prod, argmax, argmin, softmax } from '../../src/index.js';
import { layer_norm, group_norm } from '../../src/nn/functional/normalization.js';
import { conv2d } from '../../src/nn/functional/conv.js';
import { max_pool2d, avg_pool2d } from '../../src/nn/functional/pooling.js';
import { compile } from '../../src/tracing/compile.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';
import { eq, where, gt, clamp, pad, one_hot, index_select, cat, stack, roll, flip, cumsum, sort, topk, argsort, gather, scatter_add, scatter } from '../../src/index.js';
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
  { name: 'roll_dim1', shapes: [[3, 5]], fwd: (x) => roll(x, 2, 1) },
  { name: 'flip_dims', shapes: [[3, 4]], fwd: (x) => flip(x, [0, 1]) },
  { name: 'cumsum_dim1', shapes: [[3, 7]], fwd: (x) => cumsum(x, 1) },
  { name: 'cumsum_dim0', shapes: [[5, 3]], fwd: (x) => cumsum(x, 0) },
  { name: 'sort_last_pow2', shapes: [[3, 8]], fwd: (x) => sort(x) },
  { name: 'sort_last_nonpow2', shapes: [[4, 5]], fwd: (x) => sort(x) },
  { name: 'sort_last_desc', shapes: [[3, 6]], fwd: (x) => sort(x, -1, true) },
  { name: 'sort_dim0', shapes: [[5, 4]], fwd: (x) => sort(x, 0) },
  { name: 'topk_last', shapes: [[3, 7]], fwd: (x) => topk(x, 3)[0] },
  { name: 'topk_smallest', shapes: [[3, 7]], fwd: (x) => topk(x, 2, -1, false)[0] },
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

    it(`topk indices match eager on ${tname}`, () => {
      const x = tensor([[5, 3, 1, 4, 2, 8, 7], [-1, -5, 0, 9, 6, 3, 2]]);
      return checkIndexDiff('topk_idx', T, [x], (a) => topk(a, 4)[1]);
    });

    it(`argsort matches eager on ${tname}`, () => {
      const x = tensor([[5, 3, 1, 4, 2], [-1, -5, 0, 9, 6]]);
      return checkIndexDiff('argsort', T, [x], (a) => argsort(a));
    });

    it(`gather dim1 matches eager on ${tname}`, () => {
      const x = tensor([[10, 11, 12], [20, 21, 22]]);
      const idx = tensor([[2, 0], [1, 2]], { dtype: 'i32' });
      return checkIndexDiff('gather_d1', T, [x, idx], (a, b) => gather(a, 1, b));
    });

    it(`gather dim0 matches eager on ${tname}`, () => {
      const x = tensor([[10, 11, 12], [20, 21, 22]]);
      const idx = tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'i32' });
      return checkIndexDiff('gather_d0', T, [x, idx], (a, b) => gather(a, 0, b));
    });

    it(`scatter_add dim1 matches eager on ${tname}`, () => {
      const x = tensor([[1, 1, 1], [1, 1, 1]]);
      const idx = tensor([[0, 2], [1, 1]], { dtype: 'i32' });
      const src = tensor([[5, 6], [7, 8]]);
      return checkIndexDiff('scatter_add_d1', T, [x, idx, src], (a, b, c) => scatter_add(a, 1, b, c));
    });

    it(`scatter (overwrite) dim1 matches eager on ${tname}`, () => {
      const x = tensor([[1, 1, 1], [1, 1, 1]]);
      const idx = tensor([[0, 2], [0, 2]], { dtype: 'i32' });
      const src = tensor([[5, 6], [7, 8]]);
      return checkIndexDiff('scatter_d1', T, [x, idx, src], (a, b, c) => scatter(a, 1, b, c));
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

  it('roll shifts circularly', () => {
    const x = tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    expect(Array.from(roll(x, 1, 1).contiguous().data)).toEqual([4, 1, 2, 3, 8, 5, 6, 7]);
    expect(Array.from(roll(x, -1, 0).contiguous().data)).toEqual([5, 6, 7, 8, 1, 2, 3, 4]);
  });

  it('flip reverses along dims', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    expect(Array.from(flip(x, 1).contiguous().data)).toEqual([3, 2, 1, 6, 5, 4]);
    expect(Array.from(flip(x, [0, 1]).contiguous().data)).toEqual([6, 5, 4, 3, 2, 1]);
  });

  it('cumsum accumulates prefix sums', () => {
    const x = tensor([[1, 2, 3, 4, 5]]);
    expect(Array.from(cumsum(x, 1).contiguous().data)).toEqual([1, 3, 6, 10, 15]);
    const y = tensor([[1, 1], [2, 2], [3, 3]]);
    expect(Array.from(cumsum(y, 0).contiguous().data)).toEqual([1, 1, 3, 3, 6, 6]);
  });

  const jsAsc = (a) => [...a].sort((p, q) => p - q);
  const jsDesc = (a) => [...a].sort((p, q) => q - p);

  it('sort ascending matches Array.prototype.sort (power-of-two length)', () => {
    const rows = [[3, 1, 4, 1, 5, 9, 2, 6], [-2, -2, 0, 7, 7, 1, -8, 3]];
    const out = Array.from(sort(tensor(rows)).contiguous().data);
    const exp = [...jsAsc(rows[0]), ...jsAsc(rows[1])];
    expect(out).toEqual(exp);
  });

  it('sort ascending matches Array.prototype.sort (non-power-of-two length)', () => {
    const rows = [[5, 3, 1, 4, 2], [-1, -5, 0, 9, 9], [2, 2, 2, 1, 3]];
    const out = Array.from(sort(tensor(rows)).contiguous().data);
    const exp = [...jsAsc(rows[0]), ...jsAsc(rows[1]), ...jsAsc(rows[2])];
    expect(out).toEqual(exp);
  });

  it('sort descending matches reversed Array.prototype.sort', () => {
    const rows = [[5, 3, 1, 4, 2, 6], [-1, -5, 0, 9, 9, -3]];
    const out = Array.from(sort(tensor(rows), -1, true).contiguous().data);
    const exp = [...jsDesc(rows[0]), ...jsDesc(rows[1])];
    expect(out).toEqual(exp);
  });

  it('sort along a non-last dim sorts each column', () => {
    const x = tensor([[3, 6], [1, 4], [2, 5]]);
    const out = Array.from(sort(x, 0).contiguous().data);
    expect(out).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it('topk largest returns sorted top-k values', () => {
    const x = tensor([[5, 3, 1, 4, 2], [-1, -5, 0, 9, 9]]);
    const [tk] = topk(x, 3);
    expect(tk.shape).toEqual([2, 3]);
    expect(Array.from(tk.contiguous().data)).toEqual([5, 4, 3, 9, 9, 0]);
  });

  it('topk smallest returns sorted bottom-k values', () => {
    const x = tensor([[5, 3, 1, 4, 2], [-1, -5, 0, 9, 9]]);
    const [tk] = topk(x, 2, -1, false);
    expect(Array.from(tk.contiguous().data)).toEqual([1, 2, -5, -1]);
  });

  it('topk returns indices whose gather reproduces the values', () => {
    const rows = [[5, 3, 1, 4, 2], [-1, -5, 0, 7, 6]];
    const x = tensor(rows);
    const [v, idx] = topk(x, 3);
    expect(v.shape).toEqual([2, 3]);
    expect(idx.shape).toEqual([2, 3]);
    expect(idx.dtype).toBe('i32');
    const vd = Array.from(v.contiguous().data);
    const id = Array.from(idx.contiguous().data);
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        expect(rows[r][id[r * 3 + c]]).toBe(vd[r * 3 + c]);
      }
    }
    expect(id).toEqual([0, 3, 1, 3, 4, 2]);
  });

  it('topk indices handle duplicate values deterministically', () => {
    const rows = [[2, 9, 9, 1, 9]];
    const x = tensor(rows);
    const [v, idx] = topk(x, 3);
    const vd = Array.from(v.contiguous().data);
    const id = Array.from(idx.contiguous().data);
    expect(vd).toEqual([9, 9, 9]);
    expect(id).toEqual([1, 2, 4]);
    for (let c = 0; c < 3; c++) expect(rows[0][id[c]]).toBe(vd[c]);
  });

  it('argsort returns a permutation that reproduces the sort', () => {
    const rows = [[5, 3, 1, 4, 2], [-1, -5, 0, 9, 9]];
    const x = tensor(rows);
    const idx = argsort(x);
    const id = Array.from(idx.contiguous().data);
    const sorted = Array.from(sort(x).contiguous().data);
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 5; c++) expect(rows[r][id[r * 5 + c]]).toBe(sorted[r * 5 + c]);
    }
    expect(id).toEqual([2, 4, 1, 3, 0, 1, 0, 2, 3, 4]);
  });

  it('topk/argsort along a non-last dim gather correctly', () => {
    const x = tensor([[3, 6], [1, 4], [2, 5]]);
    const idx = argsort(x, 0);
    const id = Array.from(idx.contiguous().data);
    const cols = [[3, 1, 2], [6, 4, 5]];
    const sortedCols = [[1, 2, 3], [4, 5, 6]];
    for (let c = 0; c < 2; c++) for (let r = 0; r < 3; r++) {
      expect(cols[c][id[r * 2 + c]]).toBe(sortedCols[c][r]);
    }
  });

  it('gather (PyTorch semantics) picks per-element along dim', () => {
    const x = tensor([[10, 11, 12], [20, 21, 22]]);
    expect(Array.from(gather(x, 1, tensor([[2, 0], [1, 2]], { dtype: 'i32' })).contiguous().data))
      .toEqual([12, 10, 21, 22]);
    expect(Array.from(gather(x, 0, tensor([[1, 0, 1], [0, 1, 0]], { dtype: 'i32' })).contiguous().data))
      .toEqual([20, 11, 22, 10, 21, 12]);
  });

  it('gather inverts argsort (gather(self, dim, argsort) == sort)', () => {
    const x = tensor([[5, 3, 1, 4, 2], [-1, -5, 0, 9, 6]]);
    const g = gather(x, 1, argsort(x));
    expect(Array.from(g.contiguous().data)).toEqual(Array.from(sort(x).contiguous().data));
  });

  it('scatter_add accumulates src into self at indexed positions', () => {
    const base = tensor([[1, 1, 1], [1, 1, 1]]);
    const idx = tensor([[0, 2], [1, 1]], { dtype: 'i32' });
    const src = tensor([[5, 6], [7, 8]]);
    expect(Array.from(scatter_add(base, 1, idx, src).contiguous().data)).toEqual([6, 1, 7, 1, 16, 1]);
  });

  it('scatter overwrites self at indexed positions (unique indices)', () => {
    const base = tensor([[1, 1, 1], [1, 1, 1]]);
    const idx = tensor([[0, 2], [0, 2]], { dtype: 'i32' });
    const src = tensor([[5, 6], [7, 8]]);
    expect(Array.from(scatter(base, 1, idx, src).contiguous().data)).toEqual([5, 1, 6, 7, 1, 8]);
  });
});

describe('true half precision (f16/bf16): 2-byte storage + rounding', () => {
  const flat2 = (t) => {
    const a = t.toArray();
    const out = [];
    const rec = (x) => Array.isArray(x) ? x.forEach(rec) : out.push(x);
    rec(a);
    return out;
  };
  const numelOf = (s) => s.reduce((a, b) => a * b, 1);
  const refAdd = (a, b) => a.map((v, i) => v + b[i]);
  const refMul = (a, b) => a.map((v, i) => v * b[i]);
  const refSumLastDim = (a, rows, cols) => {
    const out = [];
    for (let r = 0; r < rows; r++) { let s = 0; for (let c = 0; c < cols; c++) s += a[r * cols + c]; out.push(s); }
    return out;
  };
  const refMatmul = (a, b, M, K, N) => {
    const out = new Array(M * N).fill(0);
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j]; out[i * N + j] = s; }
    return out;
  };

  it('f16 storage is exactly 2 bytes; bf16 too', () => {
    expect(tensor([1, 2, 3], { dtype: 'f16' })._impl.storage.data.BYTES_PER_ELEMENT).toBe(2);
    expect(tensor([1, 2, 3], { dtype: 'bf16' })._impl.storage.data.BYTES_PER_ELEMENT).toBe(2);
  });

  it('f16 rounds to nearest representable half on creation', () => {
    const t = tensor([1.0009765625, 1.0001, 0.5, 2049], { dtype: 'f16' });
    const v = t.toArray();
    expect(v[0]).toBe(1.0009765625);
    expect(v[1]).toBe(1.0);
    expect(v[2]).toBe(0.5);
    expect(v[3]).toBe(2048);
  });

  it('bf16 keeps ~8 bits of mantissa', () => {
    const v = tensor([1.5, 100, 1 / 3], { dtype: 'bf16' }).toArray();
    expect(v[0]).toBe(1.5);
    expect(v[1]).toBe(100);
    expect(Math.abs(v[2] - 1 / 3)).toBeLessThan(4e-3);
  });

  const TOL = { f16: 4e-3, bf16: 4e-2 };
  const HALVES = ['f16', 'bf16'];
  const run = async (dt, makeTarget, fwd, inputs, ref) => {
    const eagerOut = flat2(fwd(...inputs));
    const compiled = compile({ forward: (...a) => fwd(...a) }, inputs, { target: makeTarget() });
    const out = flat2(await compiled(...inputs));
    expect(out.length).toBe(ref.length);
    for (let i = 0; i < ref.length; i++) {
      const re = Math.abs(eagerOut[i] - ref[i]) / (1 + Math.abs(ref[i]));
      const rc = Math.abs(out[i] - ref[i]) / (1 + Math.abs(ref[i]));
      expect(re, `eager ${dt} idx ${i}: ${eagerOut[i]} vs ref ${ref[i]}`).toBeLessThan(TOL[dt]);
      expect(rc, `compiled ${dt} idx ${i}: ${out[i]} vs ref ${ref[i]}`).toBeLessThan(TOL[dt]);
    }
  };

  for (const dt of HALVES) {
    for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`${dt} add matches f32 reference on ${tname}`, async () => {
        const a = tensor([[1.5, 2.25, -0.5], [0.25, 3.5, -1.75]], { dtype: dt });
        const b = tensor([[0.5, 1.0, 2.0], [-0.25, 0.75, 1.5]], { dtype: dt });
        await run(dt, T, (x, y) => add(x, y), [a, b], refAdd(flat2(a), flat2(b)));
      });

      it(`${dt} mul matches f32 reference on ${tname}`, async () => {
        const a = tensor([[1.5, 2.25, -0.5], [0.25, 3.5, -1.75]], { dtype: dt });
        const b = tensor([[0.5, 1.0, 2.0], [-0.25, 0.75, 1.5]], { dtype: dt });
        await run(dt, T, (x, y) => mul(x, y), [a, b], refMul(flat2(a), flat2(b)));
      });

      it(`${dt} sum(dim=1) matches f32 reference on ${tname}`, async () => {
        const a = tensor([[1.5, 2.25, -0.5, 0.75], [0.25, 3.5, -1.75, 1.0]], { dtype: dt });
        await run(dt, T, (x) => sum(x, 1), [a], refSumLastDim(flat2(a), 2, 4));
      });

      it(`${dt} matmul matches f32 reference on ${tname}`, async () => {
        const a = tensor([[1, 2, 3], [4, 5, 6]], { dtype: dt });
        const b = tensor([[0.5, 1], [1.5, -1], [2, 0.5]], { dtype: dt });
        await run(dt, T, (x, y) => matmul(x, y), [a, b], refMatmul(flat2(a), flat2(b), 2, 3, 2));
      });
    }
  }

  it('f16 round-trips half-rounded values exactly through compute (cpu == wasm bit-exact)', async () => {
    const a = tensor([[1.5, 2.25, 0.5, -3.0], [10.5, -0.25, 7.75, 1024.5]], { dtype: 'f16' });
    const b = tensor([[0.5, 0.75, -1.5, 2.0], [0.25, 4.5, -2.25, 0.5]], { dtype: 'f16' });
    const fc = await compile({ forward: (x, y) => mul(add(x, y), y) }, [a, b], { target: CPUTarget() })(a, b);
    const fw = await compile({ forward: (x, y) => mul(add(x, y), y) }, [a, b], { target: WasmTarget() })(a, b);
    expect(Array.from(fc._impl.storage.data)).toEqual(Array.from(fw._impl.storage.data));
    expect(fc._impl.storage.data.BYTES_PER_ELEMENT).toBe(2);
  });
});

describe('i64 (BigInt host marshalling)', () => {
  const flat2 = (t) => {
    const out = [];
    const rec = (x) => Array.isArray(x) ? x.forEach(rec) : out.push(x);
    rec(t.toArray());
    return out;
  };

  it('tensor(...) accepts JS numbers for i64 and stores 8-byte BigInt64Array', () => {
    const t = tensor([[3, 5, 7], [2, 4, 6]], { dtype: 'i64' });
    expect(t._impl.storage.data.BYTES_PER_ELEMENT).toBe(8);
    expect(t._impl.storage.data instanceof BigInt64Array).toBe(true);
    expect(flat2(t)).toEqual([3, 5, 7, 2, 4, 6]);
  });

  for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`i64 add matches integer reference on ${tname}`, async () => {
      const a = tensor([[3, 5, 7], [2, 4, 6]], { dtype: 'i64' });
      const b = tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' });
      const out = await compile({ forward: (x, y) => add(x, y) }, [a, b], { target: T() })(a, b);
      expect(flat2(out)).toEqual([4, 7, 10, 6, 9, 12]);
      expect(flat2(add(a, b))).toEqual([4, 7, 10, 6, 9, 12]);
    });

    it(`i64 mul matches integer reference on ${tname}`, async () => {
      const a = tensor([[3, 5, 7], [2, 4, 6]], { dtype: 'i64' });
      const b = tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' });
      const out = await compile({ forward: (x, y) => mul(x, y) }, [a, b], { target: T() })(a, b);
      expect(flat2(out)).toEqual([3, 10, 21, 8, 20, 36]);
    });

    it(`i64 sum(dim=1) matches integer reference on ${tname}`, async () => {
      const a = tensor([[3, 5, 7], [2, 4, 6]], { dtype: 'i64' });
      const out = await compile({ forward: (x) => sum(x, 1) }, [a], { target: T() })(a);
      expect(flat2(out)).toEqual([15, 12]);
    });

    it(`i64 matmul matches integer reference on ${tname}`, async () => {
      const m1 = tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' });
      const m2 = tensor([[1, 0], [0, 1], [1, 1]], { dtype: 'i64' });
      const out = await compile({ forward: (x, y) => matmul(x, y) }, [m1, m2], { target: T() })(m1, m2);
      expect(flat2(out)).toEqual([4, 5, 10, 11]);
    });

    it(`i64 preserves precision above 2^53 on ${tname}`, async () => {
      const a = tensor([9007199254740992n, 4503599627370496n], { dtype: 'i64' });
      const b = tensor([1n, 4503599627370496n], { dtype: 'i64' });
      const out = await compile({ forward: (x, y) => add(x, y) }, [a, b], { target: T() })(a, b);
      expect(Array.from(out._impl.storage.data)).toEqual([9007199254740993n, 9007199254740992n]);
    });
  }
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

describe('i32 division edge cases: div-by-zero and INT_MIN/-1 match eager + JS oracle', () => {
  const INT_MIN = -2147483648;
  const dividend = [10, -10, 0, 7, -7, 5, INT_MIN, 100, INT_MIN];
  const divisor = [0, 0, 0, 2, 2, -2, -1, 0, 3];
  const oracle = dividend.map((a, i) => (a / divisor[i]) | 0);

  for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`i32 div with zero/overflow divisors on ${tname}`, async () => {
      const a = tensor([dividend], { dtype: 'i32' });
      const b = tensor([divisor], { dtype: 'i32' });
      const eager = Array.from(div(a, b).contiguous().data);
      expect(eager).toEqual(oracle);
      const compiled = compile({ forward: (x, y) => div(x, y) }, [a, b], { target: T() });
      const out = Array.from((await compiled(a, b)).contiguous().data);
      expect(out).toEqual(oracle);
    });
  }
});

describe('dynamic shapes: compile once, run on multiple concrete shapes', () => {
  const mkShape = (shape) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const flat = [];
    for (let i = 0; i < n; i++) flat.push(((i * 7) % 13) - 6 + 0.25);
    return tensor(nest(flat, shape));
  };
  const L = (t) => Array.from(t.contiguous().data);

  async function runDyn(name, fwd, dynspec, shapeLists, makeTarget) {
    const compiled = compile({ forward: (...a) => fwd(...a) }, shapeLists[0].map(mkShape), { target: makeTarget(), dynamic_shapes: dynspec });
    for (const shapes of shapeLists) {
      const ins = shapes.map(mkShape);
      const eager = L(fwd(...ins));
      const out = L(await compiled(...ins));
      expect(out.length, `${name} ${JSON.stringify(shapes)} length`).toBe(eager.length);
      for (let i = 0; i < eager.length; i++) {
        const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
        expect(relErr, `${name} ${JSON.stringify(shapes)} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(2e-3);
      }
    }
  }

  const S = (...d) => new Set(d);
  const CASES = [
    ['relu', (x) => relu(x), [S(0)], [[[3, 4]], [[5, 4]], [[1, 4]]]],
    ['relu(mul) chain', (x) => relu(mul(x, x)), [S(0)], [[[3, 4]], [[7, 4]]]],
    ['add row-broadcast', (x, y) => add(x, y), [S(0), null], [[[3, 4], [1, 4]], [[6, 4], [1, 4]]]],
    ['add same-dyn', (x, y) => add(x, y), [S(0), S(0)], [[[3, 4], [3, 4]], [[6, 4], [6, 4]]]],
    ['sum dim1', (x) => sum(x, 1), [S(0)], [[[3, 4]], [[5, 4]]]],
    ['mean dim1', (x) => mean(x, 1), [S(0)], [[[3, 4]], [[6, 4]]]],
    ['sum dim0 (reduce dynamic axis)', (x) => sum(x, 0), [S(0)], [[[3, 4]], [[5, 4]]]],
    ['max dim1', (x) => max(x, 1), [S(0)], [[[3, 4]], [[5, 4]]]],
    ['matmul dynamic M', (x, y) => matmul(x, y), [S(0), null], [[[3, 4], [4, 2]], [[7, 4], [4, 2]]]],
    ['matmul dynamic K', (x, y) => matmul(x, y), [S(1), S(0)], [[[3, 4], [4, 2]], [[3, 6], [6, 2]]]],
    ['sum keepdim dim1', (x) => sum(x, 1, true), [S(0)], [[[3, 4]], [[6, 4]]]],
    ['mean keepdim over dynamic axis', (x) => mean(x, 0, true), [S(0)], [[[4, 3]], [[2, 3]]]],
    ['softmax dim1', (x) => softmax(x, 1), [S(0)], [[[3, 5]], [[8, 5]]]],
    ['normalize (reduce keepdim then broadcast)', (x) => div(x, sum(x, 1, true)), [S(0)], [[[3, 4]], [[5, 4]]]],
    ['reduce then elementwise', (x) => relu(sub(x, mean(x, 1, true))), [S(0)], [[[3, 4]], [[6, 4]]]],
    ['reduce-result fed to elementwise', (x) => add(sum(x, 1), sum(relu(x), 1)), [S(0)], [[[3, 4]], [[5, 4]]]],
    ['two dynamic dims (relu)', (x) => relu(x), [S(0, 1)], [[[3, 4]], [[5, 7]]]],
    ['two dynamic dims (add)', (x, y) => add(x, y), [S(0, 1), S(0, 1)], [[[3, 4], [3, 4]], [[5, 7], [5, 7]]]],
    ['transpose with dynamic dim', (x) => add(x.transpose(0, 1), x.transpose(0, 1)), [S(0)], [[[3, 4]], [[5, 4]]]],
  ];

  for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    for (const [name, fwd, dynspec, shapeLists] of CASES) {
      it(`${name} on ${tname}`, () => runDyn(name, fwd, dynspec, shapeLists, T));
    }
  }

  it('explicit oracle: sum(dim1) on a symbolic-batch kernel reused across shapes', async () => {
    const compiled = compile({ forward: (x) => sum(x, 1) }, [tensor([[1, 2, 3, 4]])], { target: CPUTarget(), dynamic_shapes: [new Set([0])] });
    expect(Array.from((await compiled(tensor([[1, 2, 3, 4], [10, 20, 30, 40]]))).contiguous().data)).toEqual([10, 100]);
    expect(Array.from((await compiled(tensor([[5, 5, 5, 5]]))).contiguous().data)).toEqual([20]);
    expect(Array.from((await compiled(tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))).contiguous().data)).toEqual([4, 8, 12]);
  });

  for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`explicit oracle: mean over a dynamic axis divides by the runtime extent on ${tname}`, async () => {
      const compiled = compile({ forward: (x) => mean(x, 0) }, [tensor([[2, 4], [6, 8]])], { target: T(), dynamic_shapes: [new Set([0])] });
      expect(Array.from((await compiled(tensor([[2, 4], [6, 8]]))).contiguous().data)).toEqual([4, 6]);
      expect(Array.from((await compiled(tensor([[3, 9], [3, 9], [3, 9], [3, 9]]))).contiguous().data)).toEqual([3, 9]);
      expect(Array.from((await compiled(tensor([[10, 20]]))).contiguous().data)).toEqual([10, 20]);
    });

    it(`explicit oracle: transpose of a dynamic dim resolves the right extent on ${tname}`, async () => {
      const compiled = compile({ forward: (x) => x.transpose(0, 1) }, [tensor([[1, 2, 3], [4, 5, 6]])], { target: T(), dynamic_shapes: [new Set([0])] });
      const out2 = await compiled(tensor([[1, 2, 3], [4, 5, 6]]));
      expect(out2.shape).toEqual([3, 2]);
      expect(Array.from(out2.contiguous().data)).toEqual([1, 4, 2, 5, 3, 6]);
      const out3 = await compiled(tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
      expect(out3.shape).toEqual([3, 3]);
      expect(Array.from(out3.contiguous().data)).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9]);
    });
  }
});
