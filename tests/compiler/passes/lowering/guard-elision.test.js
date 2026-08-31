import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';

const hasGuard = (src) => /\?/.test(src);

describe('Analyzer-driven bounds-guard elision in lowering', () => {
  it('pool2d with pad=0 drops the in-bounds guard but keeps correct values', () => {
    const func = buildFunction('p', [new TensorType([1, 1, 4, 4], ScalarType.F32)], [new TensorType([1, 1, 2, 2], ScalarType.F32)],
      (b, a) => { b.returnOp([b.pool2d(a[0], 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]); });
    const r = compileGraph(func, CPUTarget());
    expect(hasGuard(r.getSource('p'))).toBe(false);
    const out = new Float32Array(4);
    r.run('p', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), out);
    expect(Array.from(out)).toEqual([6, 8, 14, 16]);
  });

  it('pool2d with padding keeps the in-bounds guard', () => {
    const func = buildFunction('p', [new TensorType([1, 1, 3, 3], ScalarType.F32)], [new TensorType([1, 1, 2, 2], ScalarType.F32)],
      (b, a) => { b.returnOp([b.pool2d(a[0], 'max', [2, 2], [2, 2], [[0, 1], [0, 1]]).getResult(0)]); });
    const r = compileGraph(func, CPUTarget());
    expect(hasGuard(r.getSource('p'))).toBe(true);
  });

  it('a no-op pad drops the in-bounds guard and copies the input', () => {
    const func = buildFunction('p', [new TensorType([2, 3], ScalarType.F32)], [new TensorType([2, 3], ScalarType.F32)],
      (b, a) => {
        const z = b.constant(0, new TensorType([], ScalarType.F32)).getResult(0);
        b.returnOp([b._inferAndBuild('pad', [a[0], z], { low: [0, 0], high: [0, 0], interior: [0, 0] }).getResult(0)]);
      });
    const r = compileGraph(func, CPUTarget());
    expect(hasGuard(r.getSource('p'))).toBe(false);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(6);
    r.run('p', inp, out);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('a real pad keeps the guard and zero-fills the border', () => {
    const func = buildFunction('p', [new TensorType([2, 2], ScalarType.F32)], [new TensorType([3, 3], ScalarType.F32)],
      (b, a) => {
        const z = b.constant(0, new TensorType([], ScalarType.F32)).getResult(0);
        b.returnOp([b._inferAndBuild('pad', [a[0], z], { low: [1, 1], high: [0, 0], interior: [0, 0] }).getResult(0)]);
      });
    const r = compileGraph(func, CPUTarget());
    expect(hasGuard(r.getSource('p'))).toBe(true);
    const out = new Float32Array(9);
    r.run('p', new Float32Array([1, 2, 3, 4]), out);
    expect(Array.from(out)).toEqual([0, 0, 0, 0, 1, 2, 0, 3, 4]);
  });
});
