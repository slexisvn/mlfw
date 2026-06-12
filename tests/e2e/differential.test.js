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

const SPECIAL = [
  { name: 'div_self_zero', data: [0, 2, 0, 3], fwd: (x) => div(x, x) },
  { name: 'div_self_inf', data: [Infinity, 2, 3, 4], fwd: (x) => div(x, x) },
  { name: 'sub_self_inf', data: [Infinity, 1, -Infinity, 2], fwd: (x) => sub(x, x) },
  { name: 'sub_self_nan', data: [NaN, 1, 2, 3], fwd: (x) => sub(x, x) },
  { name: 'exp_log_neg', data: [-1, -2, 0.5, 1], fwd: (x) => exp(log(x)) },
  { name: 'div_self_relu', data: [-1, 2, -3, 4], fwd: (x) => { const r = relu(x); return div(r, r); } },
  { name: 'mul_zero_inf', data: [Infinity, 1, NaN, 2], fwd: (x) => mul(x, sub(x, x)) },
];

function eqIEEE(a, b) {
  if (Number.isNaN(a)) return Number.isNaN(b);
  if (!Number.isFinite(a)) return a === b;
  return Math.abs(a - b) / (1 + Math.abs(a)) < 1e-4;
}

describe('algebraic simplification is IEEE-sound: special values (eager == compiled)', () => {
  for (const prog of SPECIAL) {
    for (const [tname, makeTarget] of Object.entries(TARGETS)) {
      it(`${prog.name} on ${tname} matches eager (no unsound x-x/x÷x/exp∘log rewrite)`, async () => {
        const eager = flatten(prog.fwd(tensor(prog.data)));
        const compiled = compile({ forward: prog.fwd }, [tensor(prog.data)], { target: makeTarget() });
        const out = flatten(await compiled(tensor(prog.data)));
        expect(out.length).toBe(eager.length);
        for (let i = 0; i < eager.length; i++) {
          expect(eqIEEE(eager[i], out[i]), `${prog.name}/${tname} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBe(true);
        }
      });
    }
  }
});

const A4x3 = [[1, 2, 0.5], [3, -1, 2], [0.5, 1, -2], [2, 0, 1]];
const W3x5 = [[1, 0, 2, 1, 0.5], [0.5, 1, 0, 2, 1], [2, 1, 1, 0, 0.5]];
const BIAS5 = [0.1, 0.2, -0.3, 0.4, 0.5];
const EPILOGUE = [
  { name: 'matmul_bias', ins: [A4x3, W3x5, BIAS5], fwd: (x, w, b) => add(matmul(x, w), b) },
  { name: 'matmul_bias_relu', ins: [A4x3, W3x5, BIAS5], fwd: (x, w, b) => relu(add(matmul(x, w), b)) },
  { name: 'matmul_bias_tanh', ins: [A4x3, W3x5, BIAS5], fwd: (x, w, b) => tanh(add(matmul(x, w), b)) },
  { name: 'matmul_relu', ins: [A4x3, W3x5], fwd: (x, w) => relu(matmul(x, w)) },
  { name: 'matmul_scale_bias', ins: [A4x3, W3x5, BIAS5], fwd: (x, w, b) => add(mul(matmul(x, w), tensor([2])), b) },
  { name: 'matmul_bias_exp_neg', ins: [A4x3, W3x5, BIAS5], fwd: (x, w, b) => neg(exp(add(matmul(x, w), b))) },
];

describe('epilogue fusion (forced on) matches eager — LIR accumulator must not hoist loop-var-dependent index', () => {
  for (const prog of EPILOGUE) {
    for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`${prog.name} on ${tname} matches eager`, async () => {
        const eager = flatten(prog.fwd(...prog.ins.map((d) => tensor(d))));
        const target = makeTarget({ enableEpilogueFusion: true });
        const compiled = compile({ forward: prog.fwd }, prog.ins.map((d) => tensor(d)), { target, fusion: { strategy: 'xla', epilogue: true } });
        const out = flatten(await compiled(...prog.ins.map((d) => tensor(d))));
        expect(out.length).toBe(eager.length);
        for (let i = 0; i < eager.length; i++) {
          const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
          expect(relErr, `${prog.name}/${tname} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(3e-3);
        }
      });
    }
  }
});

const OPT_PROGRAMS = [
  { name: 'ew_chain', shapes: [[6, 8], [6, 8]], fwd: (x, y) => relu(sub(mul(x, y), tanh(x))) },
  { name: 'mm_relu', shapes: [[8, 12], [12, 5]], fwd: (x, y) => relu(matmul(x, y)) },
  { name: 'mm_chain', shapes: [[4, 6], [6, 8], [8, 3]], fwd: (x, y, z) => matmul(matmul(x, y), z) },
  { name: 'softmax', shapes: [[6, 11]], fwd: (x) => softmax(x, 1) },
  { name: 'reduce_after_mm', shapes: [[6, 5], [5, 7]], fwd: (x, y) => sum(relu(matmul(x, y)), 0) },
  { name: 'mm_softmax', shapes: [[6, 8], [8, 8]], fwd: (x, y) => softmax(matmul(x, y), 1) },
  { name: 'reshape_mm', shapes: [[2, 12], [6, 5]], fwd: (x, y) => relu(matmul(x.reshape([4, 6]), y)) },
];

const OPT_CONFIGS = {
  O0: { fusion: { enabled: false }, scheduling: { enabled: false }, optimization: { layout: false, rematerialization: false }, memory: { inplaceReuse: false } },
  O1: { fusion: { enabled: true }, memory: { inplaceReuse: true } },
  O1dom: { fusion: { enabled: true, strategy: 'dominator' }, memory: { inplaceReuse: true } },
  O1epi: { fusion: { enabled: true, epilogue: true }, memory: { inplaceReuse: true } },
  O2: { fusion: { enabled: true }, scheduling: { enabled: true }, optimization: { layout: true, rematerialization: true }, memory: { inplaceReuse: true } },
};

describe('opt-level differential: every pass-combo preserves semantics (eager == compiled, CPU + WASM)', () => {
  for (const prog of OPT_PROGRAMS) {
    for (const [tname, makeTarget] of Object.entries(TARGETS)) {
      for (const [cname, cfg] of Object.entries(OPT_CONFIGS)) {
        it(`${prog.name} on ${tname} @ ${cname} matches eager`, async () => {
          const rng = mulberry32(7000 + prog.name.length * 13 + cname.length);
          const inputs = prog.shapes.map((s) => mk(rng, s, -1, 1));
          const eager = flatten(prog.fwd(...inputs));
          const compiled = compile({ forward: (...a) => prog.fwd(...a) }, inputs, { target: makeTarget(), ...cfg });
          const out = flatten(await compiled(...inputs));
          expect(out.length).toBe(eager.length);
          for (let i = 0; i < eager.length; i++) {
            const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
            expect(relErr, `${prog.name}/${tname}/${cname} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(3e-3);
          }
        });
      }
    }
  }
});

describe("verify:'full' runs verifiers after every pass + tensor/LIR, accepts valid IR (eager == compiled)", () => {
  for (const prog of OPT_PROGRAMS) {
    for (const [tname, makeTarget] of Object.entries(TARGETS)) {
      it(`${prog.name} on ${tname} compiles clean under verify:'full' (+scheduling)`, async () => {
        const rng = mulberry32(4200 + prog.name.length * 7);
        const inputs = prog.shapes.map((s) => mk(rng, s, -1, 1));
        const eager = flatten(prog.fwd(...inputs));
        const compiled = compile({ forward: (...a) => prog.fwd(...a) }, inputs, {
          target: makeTarget(), verify: 'full', scheduling: { enabled: true },
        });
        const out = flatten(await compiled(...inputs));
        expect(out.length).toBe(eager.length);
        for (let i = 0; i < eager.length; i++) {
          const relErr = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
          expect(relErr, `${prog.name}/${tname} idx ${i}: eager=${eager[i]} compiled=${out[i]}`).toBeLessThan(3e-3);
        }
      });
    }
  }
});
