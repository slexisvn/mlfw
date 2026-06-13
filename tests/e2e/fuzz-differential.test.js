import { describe, it, expect } from 'vitest';
import { tensor, add, sub, mul, neg, abs, tanh, sigmoid, relu, gelu, matmul, softmax, sum } from '../../src/index.js';
import { compile } from '../../src/tracing/compile.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const prodOf = (s) => s.reduce((a, b) => a * b, 1);
function nest(flat, shape) { if (shape.length === 1) return flat.slice(0, shape[0]); const sub = prodOf(shape.slice(1)); return Array.from({ length: shape[0] }, (_, i) => nest(flat.slice(i * sub, (i + 1) * sub), shape.slice(1))); }
function mkInput(rng, shape) { const n = prodOf(shape); const f = []; for (let i = 0; i < n; i++) f.push(-1 + 2 * rng()); return tensor(nest(f, shape)); }
const flat = (v) => (v && typeof v.contiguous === 'function') ? Array.from(v.contiguous().data) : Array.from(v.data);
const eqShape = (a, b) => a.length === b.length && a.every((d, i) => d === b[i]);

const UNARY = { relu, sigmoid, tanh, gelu, neg, abs };
const BINARY = { add, sub, mul };

function applicable(shapes, matmuls) {
  const opts = [];
  for (let i = 0; i < shapes.length; i++) {
    for (const op of Object.keys(UNARY)) opts.push({ op, args: [i], outShape: shapes[i] });
    if (shapes[i].length === 2) {
      opts.push({ op: 'transpose', args: [i], outShape: [shapes[i][1], shapes[i][0]] });
      opts.push({ op: 'softmax', args: [i], outShape: shapes[i] });
      opts.push({ op: 'sum1', args: [i], outShape: [shapes[i][0]] });
    }
  }
  for (let i = 0; i < shapes.length; i++) {
    for (let j = 0; j < shapes.length; j++) {
      if (eqShape(shapes[i], shapes[j])) for (const op of Object.keys(BINARY)) opts.push({ op, args: [i, j], outShape: shapes[i] });
      if (matmuls < 2 && shapes[i].length === 2 && shapes[j].length === 2 && shapes[i][1] === shapes[j][0]) {
        opts.push({ op: 'matmul', args: [i, j], outShape: [shapes[i][0], shapes[j][1]] });
      }
    }
  }
  return opts;
}

function genProgram(rng, inShapes) {
  const shapes = [...inShapes];
  const instrs = [];
  const target = 4 + Math.floor(rng() * 5);
  let matmuls = 0, guard = 0;
  while (instrs.length < target && guard++ < 200) {
    const opts = applicable(shapes, matmuls);
    if (opts.length === 0) break;
    const choice = opts[Math.floor(rng() * opts.length)];
    if (choice.op === 'matmul') matmuls++;
    instrs.push(choice);
    shapes.push(choice.outShape);
  }
  return instrs;
}

function runProgram(instrs, inputs) {
  const vals = [...inputs];
  for (const ins of instrs) {
    const a = ins.args.map((k) => vals[k]);
    let r;
    if (UNARY[ins.op]) r = UNARY[ins.op](a[0]);
    else if (BINARY[ins.op]) r = BINARY[ins.op](a[0], a[1]);
    else if (ins.op === 'matmul') r = matmul(a[0], a[1]);
    else if (ins.op === 'softmax') r = softmax(a[0], 1);
    else if (ins.op === 'sum1') r = sum(a[0], 1);
    else r = a[0].transpose(0, 1);
    vals.push(r);
  }
  return vals[vals.length - 1];
}

const DIM_POOL = [4, 5, 6];
function pickInputShapes(rng) {
  const k = 2 + Math.floor(rng() * 2);
  return Array.from({ length: k }, () => [DIM_POOL[Math.floor(rng() * DIM_POOL.length)], DIM_POOL[Math.floor(rng() * DIM_POOL.length)]]);
}

const TARGETS = { cpu: CPUTarget, wasm: WasmTarget };
const PROGRAM_COUNT = 80;

describe('fuzz: random op graphs match eager (CPU + WASM)', () => {
  let opUses = 0;
  for (let p = 0; p < PROGRAM_COUNT; p++) {
    it(`program ${p} compiles and matches eager`, async () => {
      const rng = mulberry32(0x9e37 + p * 2654435761);
      const inShapes = pickInputShapes(rng);
      const instrs = genProgram(rng, inShapes);
      expect(instrs.length, 'generated non-empty program').toBeGreaterThan(0);
      opUses += instrs.length;
      const inputs = inShapes.map((s) => mkInput(rng, s));
      const fwd = (...xs) => runProgram(instrs, xs);
      const eager = flat(fwd(...inputs));
      for (const v of eager) expect(Number.isFinite(v), `program ${p} eager finite`).toBe(true);
      for (const [tname, mk] of Object.entries(TARGETS)) {
        const out = flat(await compile({ forward: fwd }, inputs, { target: mk() })(...inputs));
        expect(out.length, `${p}/${tname} length`).toBe(eager.length);
        let maxErr = 0, bad = -1;
        for (let i = 0; i < eager.length; i++) { const e = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i])); if (e > maxErr) { maxErr = e; bad = i; } }
        expect(maxErr, `${p}/${tname} idx ${bad} ops=${instrs.map((x) => x.op).join(',')}`).toBeLessThan(5e-3);
      }
    });
  }
});
