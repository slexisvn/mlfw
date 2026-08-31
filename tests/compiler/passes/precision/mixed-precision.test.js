import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { applyAutocast, MixedPrecisionPass } from '../../../../src/compiler/passes/precision/mixed_precision.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { F32 as F, F16, T } from '../../../_utils/ir_fixture.js';

const M = 4, K = 8, N = 4;

function matmulReduce(name) {
  return buildFunction(name, [T([M, K]), T([K, N])], [T([M])], (b, a) => {
    const mm = b.matmul(a[0], a[1]).getResult(0);
    const z = b.scalarConstant(0, F).getResult(0);
    b.returnOp([b.reduce(mm, z, [1], 'sum').getResult(0)]);
  });
}

function innerFunc(mod) {
  return mod.functions ? mod.functions().next().value : mod;
}

const A = new Float32Array(M * K).map((_, i) => Math.sin(i * 0.7) * 1.4);
const B = new Float32Array(K * N).map((_, i) => Math.cos(i * 0.4) * 1.2);

describe('autocast inserts f16 casts around allow-listed compute ops only', () => {
  it('wraps dot with f16 down-casts, keeps its f32 accumulator, and leaves the reduce in f32', () => {
    const mod = matmulReduce('amp');
    const func = innerFunc(mod);
    expect(applyAutocast(func, { allow: new Set(['dot']) })).toBe(true);

    const dot = [...func.ops()].find(o => o.opName === 'dot');
    expect(dot.getOperand(0).type.dtype).toBe(F16);
    expect(dot.getOperand(1).type.dtype).toBe(F16);
    expect(dot.getResult(0).type.dtype).toBe(F);
    expect(dot.getAttr('out_dtype')).toBe(F);

    const reduce = [...func.ops()].find(o => o.opName === 'reduce');
    expect(reduce.getOperand(0).type.dtype).toBe(F);

    const converts = [...func.ops()].filter(o => o.opName === 'convert');
    expect(converts.map(c => c.getResult(0).type.dtype)).toEqual([F16, F16]);
  });

  it('does nothing when no op is in the allow set', () => {
    const mod = matmulReduce('noamp');
    const func = innerFunc(mod);
    expect(applyAutocast(func, { allow: new Set() })).toBe(false);
    expect([...func.ops()].some(o => o.opName === 'convert')).toBe(false);
  });
});

describe('autocast reduces precision of allowed ops but stays numerically close', () => {
  it('matmul-in-f16 output differs from full-f32 by an f16 magnitude, not bit-identical', () => {
    const ref = new Float32Array(M);
    compileGraph(matmulReduce('ref'), CPUTarget(), {}).run('ref', A, B, ref);

    const mod = matmulReduce('amp');
    applyAutocast(innerFunc(mod), { allow: new Set(['dot']) });
    const out = new Float32Array(M);
    compileGraph(mod, CPUTarget(), {}).run('amp', A, B, out);

    let maxDiff = 0;
    for (let i = 0; i < M; i++) maxDiff = Math.max(maxDiff, Math.abs(ref[i] - out[i]));
    expect(maxDiff).toBeGreaterThan(0);
    expect(maxDiff).toBeLessThan(0.05);
  });

  it('a reduce-only graph is untouched and stays bit-identical to f32', () => {
    const mk = (name) => buildFunction(name, [T([M, N])], [T([M])], (b, a) => {
      const z = b.scalarConstant(0, F).getResult(0);
      b.returnOp([b.reduce(a[0], z, [1], 'sum').getResult(0)]);
    });
    const X = new Float32Array(M * N).map((_, i) => Math.sin(i) * 1.3);

    const ref = new Float32Array(M);
    compileGraph(mk('r0'), CPUTarget(), {}).run('r0', X, ref);

    const mod = mk('r1');
    expect(applyAutocast(innerFunc(mod), { allow: new Set(['dot', 'conv']) })).toBe(false);
    const out = new Float32Array(M);
    compileGraph(mod, CPUTarget(), {}).run('r1', X, out);
    for (let i = 0; i < M; i++) expect(out[i]).toBe(ref[i]);
  });
});

describe('MixedPrecisionPass reports CHANGED/UNCHANGED', () => {
  it('CHANGED when an allowed op is present, UNCHANGED otherwise', () => {
    const a = innerFunc(matmulReduce('p1'));
    expect(new MixedPrecisionPass({ allow: new Set(['dot']) }).run(a)).toBe(PassResult.CHANGED);

    const b = innerFunc(matmulReduce('p2'));
    expect(new MixedPrecisionPass({ allow: new Set(['softmax']) }).run(b)).toBe(PassResult.UNCHANGED);
  });
});
