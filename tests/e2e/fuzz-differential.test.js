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
