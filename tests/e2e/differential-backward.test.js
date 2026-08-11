import { describe, it, expect } from 'vitest';
import * as M from '../../src/index.js';
import { tensor, add, sub, mul, div, neg, sqrt, exp, tanh, sigmoid, relu, gelu, silu, matmul, sum } from '../../src/index.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../src/tracing/compile_backward.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';
import { mulberry32 } from '../_utils/rng.js';
import { numel, nest, randomNested, flat } from '../_utils/tensor_data.js';

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
  const datas = prog.shapes.map((s) => randomNested(rng, s, prog.lo ?? -1, prog.hi ?? 1));

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

const COMPOSITE_BWD = [
  { name: 'cat', shape: [3, 4], fwd: (x) => M.cat([x, M.mul(x, M.tensor([[2]]))], 1) },
  { name: 'cumsum', shape: [3, 5], fwd: (x) => M.cumsum(x, 1) },
  { name: 'flip', shape: [3, 4], fwd: (x) => M.flip(x, 1) },
  { name: 'roll', shape: [3, 4], fwd: (x) => M.roll(x, 1, 1) },
  { name: 'repeat', shape: [2, 3], fwd: (x) => x.repeat(2, 1) },
  { name: 'pad', shape: [3, 4], fwd: (x) => M.pad(x, [1, 0], [0, 2], 0) },
  { name: 'index_select', shape: [4, 3], fwd: (x) => M.index_select(x, 0, M.tensor([2, 0, 1], { dtype: 'i32' })) },
  { name: 'gather_d1', shape: [2, 3], fwd: (x) => M.gather(x, 1, M.tensor([[2, 0, 1], [1, 2, 0]], { dtype: 'i32' })) },
  { name: 'gather_dup', shape: [1, 4], fwd: (x) => M.gather(x, 1, M.tensor([[0, 0, 2, 1]], { dtype: 'i32' })) },
  { name: 'scatter_add_src', shape: [1, 3], fwd: (x) => M.scatter_add(M.tensor([[0, 0, 0, 0, 0]]), 1, M.tensor([[0, 2, 4]], { dtype: 'i32' }), x) },
  { name: 'sort_mul_const', shape: [3, 4], fwd: (x) => M.mul(M.sort(x), M.tensor([[2, 2, 2, 2]])) },
  { name: 'sort_add_const', shape: [3, 4], fwd: (x) => M.add(M.sort(x), M.tensor([[1, 1, 1, 1]])) },
  { name: 'sort_dim0_mul', shape: [4, 3], fwd: (x) => M.mul(M.sort(x, 0), M.tensor([[2, 2, 2]])) },
  { name: 'relu_mul_sort', shape: [3, 4], fwd: (x) => M.relu(M.mul(M.sort(x), M.tensor([[2, 2, 2, 2]]))) },
  { name: 'topk_mul_const', shape: [3, 5], fwd: (x) => M.mul(M.topk(x, 3)[0], M.tensor([[2, 2, 2]])) },
  { name: 'sort_plus_sort', shape: [3, 4], fwd: (x) => M.add(M.sort(x), M.sort(M.neg(x))) },
];

function checkCompositeBackward(prog, makeTarget) {
  const rng = mulberry32(40009 + prog.name.length * 53);
  const flatData = Array.from({ length: numel(prog.shape) }, () => -1 + 2 * rng());
  const fwd = (...a) => prog.fwd(...a);
  const ins = [tensor(nest(flatData, prog.shape))];
  const cf = compileWithBackward({ forward: fwd }, ins, { target: makeTarget() });
  const out = cf(...ins);
  const g = ones(Array.isArray(out) ? out[0].shape : out.shape);
  const compiled = cf.backward(g).map((t) => flat(t));
  const numeric = numericalGrad(fwd, [flatData.slice()], [prog.shape]);
  for (let k = 0; k < numeric[0].length; k++) {
    const relErr = Math.abs(numeric[0][k] - compiled[0][k]) / (1 + Math.abs(numeric[0][k]));
    expect(relErr, `${prog.name} grad idx ${k}: num=${numeric[0][k]} comp=${compiled[0][k]}`).toBeLessThan(5e-2);
  }
}

describe('differential backward: composite/new ops vs numerical gradient (independent oracle)', () => {
  for (const prog of COMPOSITE_BWD) {
    it(`${prog.name} backward on cpu matches numerical`, () => checkCompositeBackward(prog, CPUTarget));
    it(`${prog.name} backward on wasm matches numerical`, () => checkCompositeBackward(prog, WasmTarget));
  }
});

const AD_BWD = [
  { name: 'rsqrt', shape: [2, 4], lo: 0.5, hi: 2, fwd: (x) => M.rsqrt(x) },
  { name: 'rsqrt_of_gelu_pos', shape: [2, 3], lo: 0.6, hi: 2, fwd: (x) => M.rsqrt(M.gelu(x)) },
  { name: 'rsqrt_scaled', shape: [3, 2], lo: 0.5, hi: 2, fwd: (x) => M.mul(M.rsqrt(x), M.tensor([[2, 2]])) },
  { name: 'logsoftmax1_then_sum2_keepdim', shape: [2, 3, 2], lo: -1, hi: 1, fwd: (x) => M.sum(M.log_softmax(x, 1), 2, true) },
  { name: 'logsoftmax1_then_sum2_nokeep', shape: [2, 3, 2], lo: -1, hi: 1, fwd: (x) => M.sum(M.log_softmax(x, 1), 2, false) },
  { name: 'logsoftmax0_then_sum2', shape: [2, 3, 2], lo: -1, hi: 1, fwd: (x) => M.sum(M.log_softmax(x, 0), 2, true) },
  { name: 'logsoftmax1_then_mean0', shape: [3, 4], lo: -1, hi: 1, fwd: (x) => M.mean(M.log_softmax(x, 1), 0, true) },
];

function checkADBackward(prog, makeTarget) {
  const rng = mulberry32(60013 + prog.name.length * 47);
  const flatData = Array.from({ length: numel(prog.shape) }, () => prog.lo + (prog.hi - prog.lo) * rng());
  const fwd = (...a) => prog.fwd(...a);
  const ins = [tensor(nest(flatData, prog.shape))];
  const cf = compileWithBackward({ forward: fwd }, ins, { target: makeTarget() });
  const out = cf(...ins);
  const compiled = cf.backward(ones(Array.isArray(out) ? out[0].shape : out.shape)).map((t) => flat(t));
  const numeric = numericalGrad(fwd, [flatData.slice()], [prog.shape]);
  for (let k = 0; k < numeric[0].length; k++) {
    const relErr = Math.abs(numeric[0][k] - compiled[0][k]) / (1 + Math.abs(numeric[0][k]));
    expect(relErr, `${prog.name} grad idx ${k}: num=${numeric[0][k]} comp=${compiled[0][k]}`).toBeLessThan(5e-2);
  }
}

describe('differential backward: rsqrt VJP + reduce-after-log_softmax fusion vs numerical (independent oracle)', () => {
  for (const prog of AD_BWD) {
    it(`${prog.name} backward on cpu matches numerical`, () => checkADBackward(prog, CPUTarget));
    it(`${prog.name} backward on wasm matches numerical`, () => checkADBackward(prog, WasmTarget));
  }
});

const JOINT_BWD = [
  { name: 'joint_mul_add_broadcast_row', shapes: [[2, 3], [3]], fwd: (x, y) => M.mul(M.add(x, y), x) },
  { name: 'joint_mul_add_broadcast_col', shapes: [[2, 3], [2, 1]], fwd: (x, y) => M.mul(M.add(x, y), x) },
  { name: 'joint_mul_maximum_broadcast', shapes: [[2, 3], [3]], fwd: (x, y) => M.mul(M.maximum(x, y), x) },
  { name: 'joint_mul_scalar_reduce', shapes: [[3], [1]], lo: 0.5, hi: 2, fwd: (x, y) => M.mul(M.mean(x, 0, false), y) },
  { name: 'joint_chain_scalar_reduce', shapes: [[3], [1]], lo: 0.5, hi: 2, fwd: (x, y) => M.mul(M.silu(M.neg(M.mean(x, 0, false))), y) },
  { name: 'joint_logsoftmax_then_sum', shapes: [[2, 3, 2]], lo: -1, hi: 1, fwd: (x) => M.sum(M.log_softmax(x, 1), 2, true) },
];

function checkJointVsSeparate(prog) {
  const rng = mulberry32(70019 + prog.name.length * 37);
  const datas = prog.shapes.map((s) => Array.from({ length: numel(s) }, () => (prog.lo ?? -1) + ((prog.hi ?? 1) - (prog.lo ?? -1)) * rng()));
  const fwd = (...a) => prog.fwd(...a);
  const numeric = numericalGrad(fwd, datas.map((d) => d.slice()), prog.shapes);
  for (const mode of ['separate', 'joint']) {
    const ins = datas.map((d, i) => tensor(nest(d, prog.shapes[i])));
    const cf = compileWithBackward({ forward: fwd }, ins, { target: CPUTarget(), mode });
    const out = cf(...ins);
    const grads = cf.backward(ones((Array.isArray(out) ? out[0] : out).shape)).map((t) => flat(t));
    for (let i = 0; i < numeric.length; i++) {
      expect(grads[i].length, `${prog.name} ${mode} grad ${i} length`).toBe(numeric[i].length);
      for (let k = 0; k < numeric[i].length; k++) {
        const relErr = Math.abs(numeric[i][k] - grads[i][k]) / (1 + Math.abs(numeric[i][k]));
        expect(relErr, `${prog.name} ${mode} grad[${i}][${k}]: num=${numeric[i][k]} comp=${grads[i][k]}`).toBeLessThan(5e-2);
      }
    }
  }
}

describe('joint-mode backward: broadcast-reduce + scalar-reduce vs numerical (fusion-cycle guard)', () => {
  for (const prog of JOINT_BWD) {
    it(`${prog.name} joint and separate both match numerical`, () => checkJointVsSeparate(prog));
  }
});

const EAGER_PRIM_BWD = [
  { name: 'eager_cat_dim1', shapes: [[3, 4], [3, 2]], fwd: (x, y) => M.cat([x, y], 1) },
  { name: 'eager_cat_dim0', shapes: [[2, 5], [3, 5]], fwd: (x, y) => M.cat([x, y], 0) },
  { name: 'eager_cat_three', shapes: [[3, 2], [3, 1], [3, 3]], fwd: (x, y, z) => M.cat([x, y, z], 1) },
  { name: 'eager_cat_mixed', shapes: [[3, 4]], fwd: (x) => M.cat([x, M.mul(x, M.tensor([[2, 2, 2, 2]]))], 0) },
  { name: 'eager_clamp', shapes: [[4, 5]], lo: -1, hi: 1, fwd: (x) => M.clamp(x, -0.5, 0.5) },
  { name: 'eager_pad', shapes: [[3, 4]], fwd: (x) => M.pad(x, [1, 0], [0, 2], 0) },
  { name: 'eager_index_select_0', shapes: [[4, 3]], fwd: (x) => M.index_select(x, 0, M.tensor([2, 0, 1], { dtype: 'i32' })) },
  { name: 'eager_index_select_dup', shapes: [[4, 3]], fwd: (x) => M.index_select(x, 0, M.tensor([1, 1, 3, 0], { dtype: 'i32' })) },
  { name: 'eager_index_select_1', shapes: [[3, 5]], fwd: (x) => M.index_select(x, 1, M.tensor([4, 0, 2], { dtype: 'i32' })) },
  { name: 'eager_where', shapes: [[3, 4]], fwd: (x) => M.where(M.gt(x, M.tensor([[0, 0, 0, 0]])), x, M.neg(x)) },
  { name: 'eager_cumsum', shapes: [[3, 5]], fwd: (x) => M.cumsum(x, 1) },
  { name: 'eager_flip', shapes: [[3, 4]], fwd: (x) => M.flip(x, 1) },
  { name: 'eager_roll', shapes: [[3, 4]], fwd: (x) => M.roll(x, 1, 1) },
  { name: 'eager_repeat', shapes: [[2, 3]], fwd: (x) => x.repeat(2, 1) },
  { name: 'eager_split', shapes: [[3, 4]], fwd: (x) => { const p = x.split(2, 1); return M.add(p[0], p[1]); } },
];

function checkEagerVsNumerical(prog) {
  const rng = mulberry32(50021 + prog.name.length * 41);
  const datas = prog.shapes.map((s) => Array.from({ length: numel(s) }, () => (prog.lo ?? -1) + ((prog.hi ?? 1) - (prog.lo ?? -1)) * rng()));

  const eagerInputs = datas.map((d, i) => tensor(nest(d, prog.shapes[i]), { requiresGrad: true }));
  const eagerOut = prog.fwd(...eagerInputs);
  sum(eagerOut).backward();
  const eagerGrads = eagerInputs.map((x) => flat(x.grad));

  const numeric = numericalGrad((...a) => prog.fwd(...a), datas.map((d) => d.slice()), prog.shapes);

  for (let i = 0; i < eagerGrads.length; i++) {
    expect(eagerGrads[i].length).toBe(numeric[i].length);
    for (let k = 0; k < eagerGrads[i].length; k++) {
      const relErr = Math.abs(eagerGrads[i][k] - numeric[i][k]) / (1 + Math.abs(numeric[i][k]));
      expect(relErr, `${prog.name} grad input ${i} idx ${k}: eager=${eagerGrads[i][k]} numeric=${numeric[i][k]}`).toBeLessThan(5e-2);
    }
  }
}

describe('differential backward: eager autograd of jit/composite ops vs numerical (independent oracle)', () => {
  for (const prog of EAGER_PRIM_BWD) {
    it(`${prog.name} eager backward matches numerical`, () => checkEagerVsNumerical(prog));
  }
});

describe('eager autograd: cumsum reverse-gradient smoking-gun + explicit masked grads', () => {
  it('eager cumsum backward gives reverse-cumsum gradient, not all-ones', () => {
    const x = tensor([[1, 2, 3, 4, 5]], { requiresGrad: true });
    sum(M.cumsum(x, 1)).backward();
    expect(flat(x.grad)).toEqual([5, 4, 3, 2, 1]);
  });
  it('eager clamp passes grad only inside bounds', () => {
    const x = tensor([[0.1, -0.9, 0.3, -0.2]], { requiresGrad: true });
    sum(M.clamp(x, -0.5, 0.5)).backward();
    expect(flat(x.grad)).toEqual([1, 0, 1, 1]);
  });
  it('eager where routes grad by condition', () => {
    const x = tensor([[0.5, -0.3, 0.8, -0.2]], { requiresGrad: true });
    sum(M.where(M.gt(x, M.tensor([[0, 0, 0, 0]])), x, M.neg(x))).backward();
    expect(flat(x.grad)).toEqual([1, -1, 1, -1]);
  });
  it('eager index_select scatter-adds duplicate selected rows', () => {
    const x = tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], { requiresGrad: true });
    sum(M.index_select(x, 0, M.tensor([1, 1, 3, 0], { dtype: 'i32' }))).backward();
    expect(flat(x.grad)).toEqual([1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1]);
  });
});

describe('differential backward: clamp/where masked gradients (explicit, away from kinks)', () => {
  for (const [tname, T] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`clamp passes grad only inside bounds on ${tname}`, () => {
      const x = tensor([[0.1, -0.9, 0.3, -0.2]]);
      const cf = compileWithBackward({ forward: (a) => M.clamp(a, -0.5, 0.5) }, [x], { target: T() });
      const out = cf(x);
      expect(cf.backward(ones(out.shape)).map(flat)[0]).toEqual([1, 0, 1, 1]);
    });
    it(`where routes grad by condition on ${tname}`, () => {
      const x = tensor([[0.5, -0.3, 0.8, -0.2]]);
      const cf = compileWithBackward({ forward: (a) => M.where(M.gt(a, M.tensor([[0, 0, 0, 0]])), a, M.neg(a)) }, [x], { target: T() });
      const out = cf(x);
      expect(cf.backward(ones(out.shape)).map(flat)[0]).toEqual([1, -1, 1, -1]);
    });
  }
});
