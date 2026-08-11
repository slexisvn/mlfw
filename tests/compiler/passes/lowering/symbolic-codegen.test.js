import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, symbolicShapeProduct, DYNAMIC } from '../../../../src/compiler/ir/graph/types.js';
import { SymInt, symVarName } from '../../../../src/compiler/analysis/sym_int.js';
import { emitSymInt } from '../../../../src/backend/codegen_utils.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget, WebGPUTarget } from '../../../../src/backend/target.js';
import { RuntimeTensor } from '../../../../src/runtime/runtime.js';
import { F32 } from '../../../_utils/ir_fixture.js';

const n = SymInt.var('n');
const m = SymInt.var('m');
const p = SymInt.var('p');

function refDot(la, ra, M, K, P) {
  const out = new Float32Array(M * P);
  for (let i = 0; i < M; i++) for (let j = 0; j < P; j++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += la[i * K + k] * ra[k * P + j];
    out[i * P + j] = s;
  }
  return out;
}

describe('symbolic codegen — emitSymInt', () => {
  const fv = (v) => v.name;

  it('emits a bare variable via the shape-param naming convention', () => {
    expect(emitSymInt(n, fv, 'c')).toBe('_sym_n');
    expect(symVarName('n')).toBe('_sym_n');
  });

  it('emits products and sums with infix operators', () => {
    expect(emitSymInt(SymInt.mul(n, m), fv, 'c')).toBe('(_sym_n * _sym_m)');
    expect(emitSymInt(SymInt.add(n, 4), fv, 'c')).toBe('(_sym_n + 4)');
    expect(emitSymInt(SymInt.mul(n, 4), fv, 'wgsl')).toBe('(_sym_n * 4)');
  });

  it('emits ceildiv and integer division per dialect', () => {
    expect(emitSymInt(SymInt.ceilDiv(n, 8), fv, 'c')).toBe('((_sym_n + 8 - 1) / 8)');
    expect(emitSymInt(SymInt.ceilDiv(n, 8), fv, 'js')).toBe('(((_sym_n + 8 - 1) / 8) | 0)');
    expect(emitSymInt(SymInt.div(n, m), fv, 'js')).toBe('((_sym_n / _sym_m) | 0)');
    expect(emitSymInt(SymInt.div(n, m), fv, 'c')).toBe('(_sym_n / _sym_m)');
  });

  it('formats variables through the backend formatter', () => {
    const wgsl = (v) => `i32(_shapes.${v.name})`;
    expect(emitSymInt(SymInt.mul(n, m), wgsl, 'wgsl')).toBe('(i32(_shapes._sym_n) * i32(_shapes._sym_m))');
  });

  it('rejects compound symbolic expressions on the WASM dialect', () => {
    expect(() => emitSymInt(SymInt.mul(n, m), fv, 'wat')).toThrow(/WASM/);
    expect(emitSymInt(n, fv, 'wat')).toBe('_sym_n');
  });
});

describe('symbolic codegen — symbolicShapeProduct', () => {
  it('returns a number for fully static shapes', () => {
    expect(symbolicShapeProduct([3, 4])).toBe(12);
    expect(symbolicShapeProduct([])).toBe(1);
  });

  it('returns a SymInt product when a dim is symbolic', () => {
    const prod = symbolicShapeProduct([n, 4]);
    expect(prod).toBeInstanceOf(SymInt);
    expect(SymInt.equals(prod, SymInt.mul(n, 4))).toBe(true);
  });

  it('falls back to DYNAMIC when a dim is opaque-dynamic', () => {
    expect(symbolicShapeProduct([DYNAMIC, 4])).toBe(DYNAMIC);
  });

  it('TensorType.symbolicNumel mirrors symbolicShapeProduct', () => {
    const t = new TensorType([n, m], F32);
    expect(SymInt.equals(t.symbolicNumel(), SymInt.mul(n, m))).toBe(true);
    expect(new TensorType([2, 3], F32).symbolicNumel()).toBe(6);
  });
});

describe('symbolic codegen — CPU execution', () => {
  it('elementwise add with a symbolic leading dim runs at multiple concrete sizes', () => {
    const t = new TensorType([n, 4], F32);
    const f = buildFunction('add_sym', [t, t], [t], (b, [x, y]) =>
      b.returnOp([b.add(x, y).getResult(0)]));
    const res = compileGraph(f, CPUTarget());

    const src = res.getSource('add_sym');
    expect(src).toMatch(/_sym_n/);
    expect(src).toMatch(/i0_\d+ < _sym_n/);

    for (const N of [3, 7]) {
      const xa = new Float32Array(N * 4).map((_, i) => i + 1);
      const ya = new Float32Array(N * 4).map((_, i) => (i + 1) * 2);
      const out = new Float32Array(N * 4);
      res.run('add_sym',
        RuntimeTensor.fromArray(xa, [N, 4], 'f32'),
        RuntimeTensor.fromArray(ya, [N, 4], 'f32'),
        new RuntimeTensor(out, [N, 4], 'f32'));
      for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(xa[i] + ya[i], 5);
    }
  });

  it('matmul with symbolic M and N matches a static reference', () => {
    const K = 8;
    const lhs = new TensorType([m, K], F32);
    const rhs = new TensorType([K, p], F32);
    const outT = new TensorType([m, p], F32);
    const f = buildFunction('dot_sym', [lhs, rhs], [outT], (b, [a, c]) =>
      b.returnOp([b.dot(a, c, [1], [0]).getResult(0)]));
    const res = compileGraph(f, CPUTarget());

    const src = res.getSource('dot_sym');
    expect(src).toMatch(/_sym_m/);
    expect(src).toMatch(/_sym_p/);
    expect(src).toMatch(/_sym_p\) \+ /);

    for (const [M, P] of [[2, 3], [4, 5]]) {
      const la = new Float32Array(M * K).map((_, i) => (i % 7) + 1);
      const ra = new Float32Array(K * P).map((_, i) => (i % 5) + 1);
      const out = new Float32Array(M * P);
      res.run('dot_sym',
        RuntimeTensor.fromArray(la, [M, K], 'f32'),
        RuntimeTensor.fromArray(ra, [K, P], 'f32'),
        new RuntimeTensor(out, [M, P], 'f32'));
      const ref = refDot(la, ra, M, K, P);
      for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(ref[i], 3);
    }
  });
});

describe('symbolic codegen — GPU source generation', () => {
  it('CUDA kernel declares and uses int shape params', () => {
    const lhs = new TensorType([m, 8], F32);
    const rhs = new TensorType([8, p], F32);
    const outT = new TensorType([m, p], F32);
    const f = buildFunction('dot_sym', [lhs, rhs], [outT], (b, [a, c]) =>
      b.returnOp([b.dot(a, c, [1], [0]).getResult(0)]));
    const src = compileGraph(f, CUDATarget()).getSource('dot_sym');
    expect(src).toMatch(/__global__ void dot_sym\([^)]*int _sym_m[^)]*int _sym_p/);
    expect(src).toMatch(/< _sym_m/);
    expect(src).toMatch(/< _sym_p/);
  });

  it('WebGPU kernel exposes shape params via the uniform struct and runs serially', () => {
    const t = new TensorType([n, 4], F32);
    const f = buildFunction('add_sym', [t, t], [t], (b, [x, y]) =>
      b.returnOp([b.add(x, y).getResult(0)]));
    const src = compileGraph(f, WebGPUTarget()).getSource('add_sym');
    expect(src).toMatch(/struct ShapeParams/);
    expect(src).toMatch(/_sym_n: u32/);
    expect(src).toMatch(/i32\(_shapes\._sym_n\)/);
  });
});

describe('symbolic codegen — static regression', () => {
  it('fully static graphs emit no shape params (byte-identical path)', () => {
    const t = new TensorType([3, 4], F32);
    const f = buildFunction('add_static', [t, t], [t], (b, [x, y]) =>
      b.returnOp([b.add(x, y).getResult(0)]));
    const src = compileGraph(f, CPUTarget()).getSource('add_static');
    expect(src).not.toMatch(/_sym_/);
    expect(src).toMatch(/< 3/);
    expect(src).toMatch(/< 4/);
  });
});
