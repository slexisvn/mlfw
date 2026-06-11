import { describe, it, expect } from 'vitest';
import * as M from '../../src/index.js';
import { compile } from '../../src/tracing/compile.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';

function rng32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const ri = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
const pick = (r, arr) => arr[Math.floor(r() * arr.length)];
const numel = (s) => s.reduce((a, b) => a * b, 1);
function nest(flat, shape) {
  if (shape.length === 1) return flat.slice(0, shape[0]);
  const sub = numel(shape.slice(1)); const o = [];
  for (let i = 0; i < shape[0]; i++) o.push(nest(flat.slice(i * sub, (i + 1) * sub), shape.slice(1)));
  return o;
}
function mkData(r, shape, kind, dtype) {
  const n = numel(shape), f = [];
  for (let i = 0; i < n; i++) {
    let v = kind === 'pos' ? 0.5 + 1.5 * r() : -1 + 2 * r();
    if (dtype === 'i32') v = ri(r, kind === 'pos' ? 1 : -5, 5);
    f.push(v);
  }
  return M.tensor(nest(f, shape), dtype ? { dtype } : undefined);
}
function logical(t) {
  return t && typeof t.contiguous === 'function' ? Array.from(t.contiguous().data) : Array.from(t.data);
}

const U_F32 = ['relu', 'sigmoid', 'tanh', 'neg', 'abs', 'gelu', 'silu'];
const U_I32 = ['neg', 'abs'];
const B_F32 = ['add', 'sub', 'mul', 'div'];
const B_I32 = ['add', 'sub', 'mul'];
const R_F32 = ['sum', 'mean', 'max', 'min'];
const R_I32 = ['sum', 'max', 'min'];

function genProgram(r, dtype) {
  const f32 = dtype !== 'i32';
  let shape = [ri(r, 2, 6), ri(r, 2, 6)];
  const inputs = [{ shape: [...shape], kind: 'normal' }];
  const steps = [];
  let idx = 1;
  const n = ri(r, 2, 6);
  for (let s = 0; s < n; s++) {
    const choices = ['unary', 'binary'];
    if (shape.length === 2 && numel(shape) > 1) choices.push('reduce');
    if (shape.length === 2) choices.push('matmul');
    const kind = pick(r, choices);
    if (kind === 'unary') {
      steps.push({ kind, op: pick(r, f32 ? U_F32 : U_I32) });
    } else if (kind === 'binary') {
      const op = pick(r, f32 ? B_F32 : B_I32);
      const rk = pick(r, shape.length === 2 ? ['same', 'rowb', 'colb'] : ['same']);
      const rs = rk === 'same' ? [...shape] : rk === 'rowb' ? [1, shape[1]] : [shape[0], 1];
      inputs.push({ shape: rs, kind: op === 'div' ? 'pos' : 'normal' });
      steps.push({ kind, op, idx: idx++ });
    } else if (kind === 'matmul') {
      const P = ri(r, 2, 6);
      inputs.push({ shape: [shape[1], P], kind: 'normal' });
      steps.push({ kind, idx: idx++ });
      shape = [shape[0], P];
    } else {
      const axis = ri(r, 0, shape.length - 1);
      const keepdim = r() < 0.5;
      steps.push({ kind: 'reduce', op: pick(r, f32 ? R_F32 : R_I32), axis, keepdim });
      if (keepdim) shape = shape.map((d, i) => i === axis ? 1 : d);
      else shape = shape.filter((_, i) => i !== axis);
    }
  }
  return { inputs, steps };
}
function applyProgram(steps, args) {
  let cur = args[0];
  for (const st of steps) {
    if (st.kind === 'unary') cur = M[st.op](cur);
    else if (st.kind === 'binary') cur = M[st.op](cur, args[st.idx]);
    else if (st.kind === 'matmul') cur = M.matmul(cur, args[st.idx]);
    else cur = M[st.op](cur, st.axis, st.keepdim);
  }
  return cur;
}

function shuffle(r, arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}
function divisorsOf(n) {
  const d = [];
  for (let i = 1; i <= n; i++) if (n % i === 0) d.push(i);
  return d;
}
function randReshape(r, n) {
  const dims = [];
  let rem = n;
  const k = ri(r, 1, 4);
  for (let i = 0; i < k - 1 && rem > 1; i++) {
    const d = pick(r, divisorsOf(rem));
    dims.push(d);
    rem = rem / d;
  }
  dims.push(rem);
  return shuffle(r, dims);
}

function genNDProgram(r) {
  let shape = Array.from({ length: ri(r, 1, 4) }, () => ri(r, 1, 5));
  const inputs = [{ shape: [...shape], kind: 'normal' }];
  const steps = [];
  let idx = 1;
  const n = ri(r, 2, 7);
  for (let s = 0; s < n; s++) {
    const rank = shape.length;
    const choices = ['unary', 'binary', 'view'];
    if (rank >= 1 && numel(shape) >= 1) choices.push('reduce');
    const kind = pick(r, choices);
    if (kind === 'unary') {
      steps.push({ kind, op: pick(r, U_F32) });
    } else if (kind === 'binary') {
      const op = pick(r, B_F32);
      const keep = ri(r, 0, rank);
      let rs = shape.slice(rank - keep).map((d) => (r() < 0.3 ? 1 : d));
      if (rs.length === 0) rs = [1];
      inputs.push({ shape: rs, kind: op === 'div' ? 'pos' : 'normal' });
      steps.push({ kind, op, idx: idx++ });
    } else if (kind === 'reduce') {
      const axis = ri(r, 0, rank - 1);
      const keepdim = r() < 0.5;
      steps.push({ kind: 'reduce', op: pick(r, R_F32), axis, keepdim });
      if (keepdim) shape = shape.map((d, i) => (i === axis ? 1 : d));
      else shape = shape.filter((_, i) => i !== axis);
    } else {
      const applicable = ['reshape', 'flatten', 'unsqueeze'];
      if (rank >= 2) applicable.push('transpose', 'permute');
      if (rank >= 1) applicable.push('slice', 'narrow', 'select');
      if (shape.some((d) => d === 1)) applicable.push('expand', 'squeeze');
      const vop = pick(r, applicable);
      if (vop === 'reshape') {
        const dims = randReshape(r, numel(shape));
        steps.push({ kind: 'view', vop: 'reshape', args: [dims] }); shape = dims;
      } else if (vop === 'flatten') {
        const dims = [numel(shape)];
        steps.push({ kind: 'view', vop: 'reshape', args: [dims] }); shape = dims;
      } else if (vop === 'transpose') {
        const d0 = ri(r, 0, rank - 1); let d1 = ri(r, 0, rank - 1); if (d1 === d0) d1 = (d1 + 1) % rank;
        steps.push({ kind: 'view', vop: 'transpose', args: [d0, d1] });
        const ns = [...shape]; const t = ns[d0]; ns[d0] = ns[d1]; ns[d1] = t; shape = ns;
      } else if (vop === 'permute') {
        const perm = shuffle(r, Array.from({ length: rank }, (_, i) => i));
        steps.push({ kind: 'view', vop: 'permute', args: [perm] });
        shape = perm.map((p) => shape[p]);
      } else if (vop === 'expand') {
        const target = shape.map((d) => (d === 1 && r() < 0.7 ? ri(r, 2, 5) : d));
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
        steps.push({ kind: 'view', vop: 'select', args: [dim, index] });
        shape = shape.filter((_, i) => i !== dim);
      } else if (vop === 'squeeze') {
        const ones = shape.map((d, i) => (d === 1 ? i : -1)).filter((i) => i >= 0);
        const dim = pick(r, ones);
        steps.push({ kind: 'view', vop: 'squeeze', args: [dim] });
        shape = shape.filter((_, i) => i !== dim);
      } else {
        const dim = ri(r, 0, shape.length);
        steps.push({ kind: 'view', vop: 'unsqueeze', args: [dim] });
        shape = [...shape.slice(0, dim), 1, ...shape.slice(dim)];
      }
    }
  }
  return { inputs, steps };
}
function applyND(steps, args) {
  let cur = args[0];
  for (const st of steps) {
    if (st.kind === 'unary') cur = M[st.op](cur);
    else if (st.kind === 'binary') cur = M[st.op](cur, args[st.idx]);
    else if (st.kind === 'reduce') cur = M[st.op](cur, st.axis, st.keepdim);
    else cur = cur[st.vop](...st.args);
  }
  return cur;
}

const N = 200;
const targets = { cpu: CPUTarget, wasm: WasmTarget };

describe('fuzz differential: eager vs compiled (seeded random op-graphs)', () => {
  it(`${N} random programs match across cpu+wasm`, async () => {
    const fails = [];
    for (let s = 0; s < N; s++) {
      const r = rng32(1000 + s * 2654435761);
      const dtype = r() < 0.25 ? 'i32' : undefined;
      const prog = genProgram(r, dtype);
      const inputs = prog.inputs.map((spec) => mkData(r, spec.shape, spec.kind, dtype));
      let eager;
      try { eager = logical(applyProgram(prog.steps, inputs)); } catch { continue; }
      if (eager.some((v) => !Number.isFinite(v))) continue;
      for (const [tn, T] of Object.entries(targets)) {
        let out;
        try {
          const cf = compile({ forward: (...a) => applyProgram(prog.steps, a) }, inputs, { target: T() });
          out = logical(await cf(...inputs));
        } catch (e) {
          fails.push(`[COMPILE ${tn}] s=${s} dtype=${dtype || 'f32'}: ${e.message.split('\n')[0]}`);
          continue;
        }
        if (out.length !== eager.length) {
          fails.push(`[SHAPE ${tn}] s=${s} eager ${eager.length} vs ${out.length}`);
          continue;
        }
        for (let i = 0; i < eager.length; i++) {
          const err = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
          if (err > 3e-3) {
            fails.push(`[VALUE ${tn}] s=${s} dtype=${dtype || 'f32'} idx${i} e=${eager[i]} c=${out[i]} steps=${JSON.stringify(prog.steps)}`);
            break;
          }
        }
      }
    }
    expect(fails, fails.slice(0, 10).join('\n')).toEqual([]);
  });
});

const N_ND = 400;

describe('fuzz differential: N-D + view-ops (rank 0-4, reshape/transpose/permute/expand/slice/squeeze/unsqueeze/narrow/select)', () => {
  it(`${N_ND} random N-D view programs match across cpu+wasm`, async () => {
    const fails = [];
    let ran = 0;
    for (let s = 0; s < N_ND; s++) {
      const r = rng32(50021 + s * 2654435761);
      const prog = genNDProgram(r);
      const inputs = prog.inputs.map((spec) => mkData(r, spec.shape, spec.kind));
      let eager;
      try { eager = logical(applyND(prog.steps, inputs)); } catch { continue; }
      if (eager.some((v) => !Number.isFinite(v))) continue;
      ran++;
      for (const [tn, T] of Object.entries(targets)) {
        let out;
        try {
          const cf = compile({ forward: (...a) => applyND(prog.steps, a) }, inputs, { target: T() });
          out = logical(await cf(...inputs));
        } catch (e) {
          fails.push(`[COMPILE ${tn}] s=${s}: ${e.message.split('\n')[0]} steps=${JSON.stringify(prog.steps)}`);
          continue;
        }
        if (out.length !== eager.length) {
          fails.push(`[SHAPE ${tn}] s=${s} eager ${eager.length} vs ${out.length} steps=${JSON.stringify(prog.steps)}`);
          continue;
        }
        for (let i = 0; i < eager.length; i++) {
          const err = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
          if (err > 3e-3) {
            fails.push(`[VALUE ${tn}] s=${s} idx${i} e=${eager[i]} c=${out[i]} steps=${JSON.stringify(prog.steps)}`);
            break;
          }
        }
      }
    }
    expect(ran, 'too many skipped programs').toBeGreaterThan(N_ND / 2);
    expect(fails, fails.slice(0, 8).join('\n')).toEqual([]);
  });
});
