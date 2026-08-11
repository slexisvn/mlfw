import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../../src/backend/target.js';
import { F32 } from '../../../_utils/ir_fixture.js';


function rnd(n, seed) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.sin(i * 0.3 + seed) * 0.5;
  return a;
}

function matmul(N) {
  return {
    name: 'mm', outShape: [N, N],
    inT: [new TensorType([N, N], F32), new TensorType([N, N], F32)],
    build: (b, a) => b.returnOp([b.matmul(a[0], a[1]).getResult(0)]),
  };
}
function reduction(R, C) {
  return {
    name: 'red', outShape: [R],
    inT: [new TensorType([R, C], F32), new TensorType([], F32)],
    build: (b, a) => b.returnOp([b.reduce(a[0], a[1], [1], 'sum').getResult(0)]),
  };
}
function conv() {
  return {
    name: 'conv', outShape: [1, 16, 30, 30],
    inT: [new TensorType([1, 8, 32, 32], F32), new TensorType([16, 8, 3, 3], F32)],
    build: (b, a) => b.returnOp([b.conv(a[0], a[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]),
  };
}

const prod = (s) => s.reduce((a, b) => a * b, 1);

function runSpec(spec, target, opts) {
  const mk = () => buildFunction(spec.name, spec.inT, [new TensorType(spec.outShape, F32)], spec.build);
  const r = compileGraph(mk(), target, opts);
  const ins = spec.inT.map((t, i) => rnd(Math.max(1, prod(t.shape)), i + 1));
  const out = new Float32Array(prod(spec.outShape));
  r.run(spec.name, ...ins, out);
  return out;
}

describe('schedule policy preserves results at realistic sizes', () => {
  const specs = [
    ['matmul 128', matmul(128)],
    ['matmul 96', matmul(96)],
    ['matmul 100', matmul(100)],
    ['matmul 64', matmul(64)],
    ['reduction 1024x512', reduction(1024, 512)],
    ['reduction 1000x300', reduction(1000, 300)],
    ['conv', conv()],
  ];
  for (const [label, spec] of specs) {
    for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`${label} scheduled matches baseline on ${tname}`, () => {
        const base = runSpec(spec, makeTarget(), {});
        const sched = runSpec(spec, makeTarget(), { scheduling: { enabled: true } });
        expect(sched.length).toBe(base.length);
        let maxErr = 0;
        for (let i = 0; i < base.length; i++) maxErr = Math.max(maxErr, Math.abs(base[i] - sched[i]));
        expect(maxErr).toBeLessThan(1e-3);
      });
    }
  }
});
