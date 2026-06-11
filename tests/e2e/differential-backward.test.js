import { describe, it, expect } from 'vitest';
import * as M from '../../src/index.js';
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

const ri = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
const pick = (r, arr) => arr[Math.floor(r() * arr.length)];
function shuffle(r, arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
  return a;
}
function divisorsOf(n) { const d = []; for (let i = 1; i <= n; i++) if (n % i === 0) d.push(i); return d; }
function randReshape(r, n) {
  const dims = []; let rem = n; const k = ri(r, 1, 3);
  for (let i = 0; i < k - 1 && rem > 1; i++) { const d = pick(r, divisorsOf(rem)); dims.push(d); rem = rem / d; }
  dims.push(rem); return shuffle(r, dims);
}

const GU = ['sigmoid', 'tanh', 'gelu', 'silu', 'neg'];
const GB = ['add', 'sub', 'mul'];
const GR = ['sum', 'mean'];

function genGradProgram(r) {
  let shape = Array.from({ length: ri(r, 1, 3) }, () => ri(r, 2, 4));
  const inputs = [{ shape: [...shape] }];
  const steps = []; let idx = 1;
  const n = ri(r, 2, 4);
  for (let s = 0; s < n; s++) {
    const rank = shape.length;
    const choices = ['unary', 'binary', 'view'];
    if (rank >= 1 && numel(shape) >= 1) choices.push('reduce');
    const kind = pick(r, choices);
    if (kind === 'unary') {
      steps.push({ kind, op: pick(r, GU) });
    } else if (kind === 'binary') {
      const op = pick(r, GB);
      const keep = ri(r, 0, rank);
      let rs = shape.slice(rank - keep).map((d) => (r() < 0.3 ? 1 : d));
      if (rs.length === 0) rs = [1];
      inputs.push({ shape: rs });
      steps.push({ kind, op, idx: idx++ });
    } else if (kind === 'reduce') {
      const axis = ri(r, 0, rank - 1);
      const keepdim = r() < 0.5;
      steps.push({ kind: 'reduce', op: pick(r, GR), axis, keepdim });
      shape = keepdim ? shape.map((d, i) => (i === axis ? 1 : d)) : shape.filter((_, i) => i !== axis);
    } else {
      const applicable = ['reshape', 'unsqueeze'];
      if (rank >= 2) applicable.push('transpose', 'permute');
      if (rank >= 1) applicable.push('slice', 'narrow', 'select');
      if (shape.some((d) => d === 1)) applicable.push('expand', 'squeeze');
      const vop = pick(r, applicable);
      if (vop === 'reshape') {
        const dims = randReshape(r, numel(shape)); steps.push({ kind: 'view', vop: 'reshape', args: [dims] }); shape = dims;
      } else if (vop === 'transpose') {
        const d0 = ri(r, 0, rank - 1); let d1 = ri(r, 0, rank - 1); if (d1 === d0) d1 = (d1 + 1) % rank;
        steps.push({ kind: 'view', vop: 'transpose', args: [d0, d1] });
        const ns = [...shape]; const t = ns[d0]; ns[d0] = ns[d1]; ns[d1] = t; shape = ns;
      } else if (vop === 'permute') {
        const perm = shuffle(r, Array.from({ length: rank }, (_, i) => i));
        steps.push({ kind: 'view', vop: 'permute', args: [perm] }); shape = perm.map((p) => shape[p]);
      } else if (vop === 'expand') {
        const target = shape.map((d) => (d === 1 && r() < 0.7 ? ri(r, 2, 4) : d));
        steps.push({ kind: 'view', vop: 'expand', args: [target] }); shape = target;
      } else if (vop === 'slice') {
        const dim = ri(r, 0, rank - 1); const len = shape[dim];
        const start = ri(r, 0, len - 1); const step = ri(r, 1, 2); const end = ri(r, start + 1, len);
        steps.push({ kind: 'view', vop: 'slice', args: [dim, start, end, step] });
        shape = [...shape]; shape[dim] = Math.ceil((end - start) / step);
      } else if (vop === 'narrow') {
        const dim = ri(r, 0, rank - 1); const len = shape[dim];
        const start = ri(r, 0, len - 1); const length = ri(r, 1, len - start);
        steps.push({ kind: 'view', vop: 'narrow', args: [dim, start, length] });
        shape = [...shape]; shape[dim] = length;
      } else if (vop === 'select') {
        const dim = ri(r, 0, rank - 1); const index = ri(r, 0, shape[dim] - 1);
        steps.push({ kind: 'view', vop: 'select', args: [dim, index] }); shape = shape.filter((_, i) => i !== dim);
      } else if (vop === 'squeeze') {
        const ones2 = shape.map((d, i) => (d === 1 ? i : -1)).filter((i) => i >= 0);
        const dim = pick(r, ones2); steps.push({ kind: 'view', vop: 'squeeze', args: [dim] }); shape = shape.filter((_, i) => i !== dim);
      } else {
        const dim = ri(r, 0, shape.length); steps.push({ kind: 'view', vop: 'unsqueeze', args: [dim] });
        shape = [...shape.slice(0, dim), 1, ...shape.slice(dim)];
      }
    }
  }
  return { inputs, steps };
}
function applyGrad(steps, args) {
  let cur = args[0];
  for (const st of steps) {
    if (st.kind === 'unary') cur = M[st.op](cur);
    else if (st.kind === 'binary') cur = M[st.op](cur, args[st.idx]);
    else if (st.kind === 'reduce') cur = M[st.op](cur, st.axis, st.keepdim);
    else cur = cur[st.vop](...st.args);
  }
  return cur;
}
function numericalGrad(fwd, datas, shapes) {
  const eps = 1e-3;
  const grads = datas.map((d) => d.map(() => 0));
  const sumFwd = (ds) => {
    const ins = ds.map((d, i) => tensor(nest(d, shapes[i])));
    return flat(sum(fwd(...ins))).reduce((a, b) => a + b, 0);
  };
  for (let i = 0; i < datas.length; i++) {
    for (let j = 0; j < datas[i].length; j++) {
      const o = datas[i][j];
      datas[i][j] = o + eps; const lp = sumFwd(datas);
      datas[i][j] = o - eps; const lm = sumFwd(datas);
      datas[i][j] = o; grads[i][j] = (lp - lm) / (2 * eps);
    }
  }
  return grads;
}

describe('fuzz backward: compiled gradients vs numerical finite-difference (independent oracle)', () => {
  it('200 random differentiable programs match numerical gradients', () => {
    const fails = [];
    let ran = 0;
    for (let s = 0; s < 200; s++) {
      const r = mulberry32(70001 + s * 2654435761);
      const prog = genGradProgram(r);
      const shapes = prog.inputs.map((spec) => spec.shape);
      const datas = shapes.map((sh) => Array.from({ length: numel(sh) }, () => -1 + 2 * r()));
      const fwd = (...a) => applyGrad(prog.steps, a);
      let numeric, compiledGrads;
      try {
        const ins = datas.map((d, i) => tensor(nest(d, shapes[i])));
        const cf = compileWithBackward({ forward: (...a) => fwd(...a) }, ins, { target: CPUTarget() });
        const out = cf(...ins);
        const fwdFlat = flat(Array.isArray(out) ? out[0] : out);
        if (fwdFlat.some((v) => !Number.isFinite(v))) continue;
        const g = ones(Array.isArray(out) ? out[0].shape : out.shape);
        compiledGrads = cf.backward(g).map((t) => flat(t));
        numeric = numericalGrad(fwd, datas, shapes);
      } catch (e) {
        fails.push(`s=${s}: ${e.message.split('\n')[0]} steps=${JSON.stringify(prog.steps)}`);
        continue;
      }
      ran++;
      for (let i = 0; i < numeric.length; i++) {
        for (let k = 0; k < numeric[i].length; k++) {
          const n = numeric[i][k]; const c = compiledGrads[i]?.[k];
          const relErr = Math.abs(n - c) / (1 + Math.abs(n));
          if (!(relErr < 5e-2)) {
            fails.push(`s=${s} grad[${i}][${k}] num=${n} comp=${c} steps=${JSON.stringify(prog.steps)}`);
            break;
          }
        }
      }
    }
    expect(ran, 'too many skipped programs').toBeGreaterThan(100);
    expect(fails, fails.slice(0, 8).join('\n')).toEqual([]);
  });
});

const SLICE_BWD = [
  { name: 'slice_step2_dim1', shapes: [[3, 6]], fwd: (x) => tanh(x.slice(1, 0, 6, 2)) },
  { name: 'slice_step2_dim0', shapes: [[6, 3]], fwd: (x) => sigmoid(x.slice(0, 1, 6, 2)) },
  { name: 'slice_step3', shapes: [[2, 9]], fwd: (x) => x.slice(1, 0, 9, 3) },
  { name: 'slice_then_matmul', shapes: [[4, 6], [3, 5]], fwd: (x, y) => matmul(x.slice(1, 0, 6, 2), y) },
];

describe('differential backward: strided slice gradients vs eager autograd', () => {
  for (const prog of SLICE_BWD) {
    it(`${prog.name} backward on cpu matches eager`, () => checkBackward(prog, CPUTarget));
    it(`${prog.name} backward on wasm matches eager`, () => checkBackward(prog, WasmTarget));
  }
});

const MULTI_OUT_FUSION_BWD = [
  { name: 'sum_sub_mean_reshape', shapes: [[3], [1]], fwd: (x, y) => M.mean(M.sub(M.sum(x, 0, true), y), 0, false).reshape([1]) },
  { name: 'mean_add_mul_mul', shapes: [[3, 4], [3, 1], [3, 4], [3, 4]], fwd: (x, y, z, w) => M.mul(M.mul(M.add(M.mean(x, 1, true), y), z), w) },
  { name: 'sum_keepdim_broadcast_sub', shapes: [[4, 5]], fwd: (x) => M.sub(x, M.sum(x, 1, true)) },
];

describe('differential backward: multi-output fusion (different-shape results) gradients vs eager autograd', () => {
  for (const prog of MULTI_OUT_FUSION_BWD) {
    it(`${prog.name} backward on cpu matches eager`, () => checkBackward(prog, CPUTarget));
    it(`${prog.name} backward on wasm matches eager`, () => checkBackward(prog, WasmTarget));
  }
});
