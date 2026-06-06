import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget(), opts);
}

describe('broadcast', () => {
  it('scalar broadcast to vector then mul', () => {
    const v = new TensorType([4], ScalarType.F32);
    const func = buildFunction('scale', [v], [v], (b, args) => {
      const c = b.scalarConstant(3, ScalarType.F32);
      const bc = b.broadcast(c.getResult(0), [4], []);
      b.returnOp([b.mul(args[0], bc.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([2, 4, 6, 8]);
    const out = new Float32Array(4);
    r.run('scale', inp, out);
    expect(Array.from(out)).toEqual([6, 12, 18, 24]);
  });

  it('vector broadcast to matrix (row-wise)', () => {
    const v = new TensorType([3], ScalarType.F32);
    const m = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('row_bc', [m, v], [m], (b, args) => {
      const bc = b.broadcast(args[1], [2, 3], [1]);
      b.returnOp([b.add(args[0], bc.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const mat = new Float32Array([1, 2, 3, 4, 5, 6]);
    const vec = new Float32Array([10, 20, 30]);
    const out = new Float32Array(6);
    r.run('row_bc', mat, vec, out);
    expect(Array.from(out)).toEqual([11, 22, 33, 14, 25, 36]);
  });
});

describe('reshape', () => {
  it('2x3 → 6', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([6], ScalarType.F32);
    const func = buildFunction('flat', [a], [b], (bb, args) => {
      bb.returnOp([bb.reshape(args[0], [6]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(6);
    r.run('flat', inp, out);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('6 → 2x3', () => {
    const a = new TensorType([6], ScalarType.F32);
    const b = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('unflt', [a], [b], (bb, args) => {
      bb.returnOp([bb.reshape(args[0], [2, 3]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(6);
    r.run('unflt', inp, out);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('reshape then elementwise', () => {
    const a = new TensorType([6], ScalarType.F32);
    const b = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('resh_neg', [a], [b], (bb, args) => {
      const r = bb.reshape(args[0], [2, 3]);
      bb.returnOp([bb.neg(r.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, -2, 3, -4, 5, -6]);
    const out = new Float32Array(6);
    r.run('resh_neg', inp, out);
    expect(Array.from(out)).toEqual([-1, 2, -3, 4, -5, 6]);
  });
});

describe('codegen quality — identity reshape', () => {
  it('same-shape reshape has no modulo or division in index arithmetic', () => {
    const a = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('id_resh', [a], [a], (bb, args) => {
      bb.returnOp([bb.reshape(args[0], [4, 8]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('id_resh');
    expect(src).not.toMatch(/%/);
    expect(src).not.toMatch(/Math\.trunc/);
  });

  it('identity reshape copies data unchanged', () => {
    const a = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('id_resh_v', [a], [a], (bb, args) => {
      bb.returnOp([bb.reshape(args[0], [3, 4]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const out = new Float32Array(12);
    r.run('id_resh_v', inp, out);
    expect(Array.from(out)).toEqual(Array.from(inp));
  });
});

describe('codegen quality — reshape index simplification', () => {
  it('reshape to same-rank different-shape uses simplified index arithmetic', () => {
    const a = new TensorType([12], ScalarType.F32);
    const b = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('resh_simp', [a], [b], (bb, args) => {
      bb.returnOp([bb.reshape(args[0], [3, 4]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('resh_simp');
    expect(src).not.toMatch(/% 1\b/);
  });
});

describe('transpose', () => {
  it('2x3 → 3x2', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([3, 2], ScalarType.F32);
    const func = buildFunction('tr', [a], [b], (bb, args) => {
      bb.returnOp([bb.transpose(args[0], [1, 0]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(6);
    r.run('tr', inp, out);
    expect(Array.from(out)).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it('transpose then matmul (A^T @ B)', () => {
    const a = new TensorType([3, 2], ScalarType.F32);
    const bt = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([3, 4], ScalarType.F32);
    const out = new TensorType([2, 4], ScalarType.F32);

    const func = buildFunction('trmm', [a, b], [out], (bb, args) => {
      const at = bb.transpose(args[0], [1, 0]);
      bb.returnOp([bb.matmul(at.getResult(0), args[1]).getResult(0)]);
    });

    const r = compile(func);
    const aData = new Float32Array([1, 0, 0, 1, 0, 0]);
    const bData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const c = new Float32Array(8);
    r.run('trmm', aData, bData, c);
    expect(Array.from(c)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

describe('slice', () => {
  it('1D slice [1:4] from length-6', () => {
    const a = new TensorType([6], ScalarType.F32);
    const b = new TensorType([3], ScalarType.F32);
    const func = buildFunction('sl1d', [a], [b], (bb, args) => {
      bb.returnOp([bb.slice(args[0], [1], [4]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([10, 20, 30, 40, 50, 60]);
    const out = new Float32Array(3);
    r.run('sl1d', inp, out);
    expect(Array.from(out)).toEqual([20, 30, 40]);
  });

  it('2D slice extracts submatrix', () => {
    const a = new TensorType([4, 4], ScalarType.F32);
    const b = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('sl2d', [a], [b], (bb, args) => {
      bb.returnOp([bb.slice(args[0], [1, 1], [3, 3]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const out = new Float32Array(4);
    r.run('sl2d', inp, out);
    expect(Array.from(out)).toEqual([6, 7, 10, 11]);
  });
});

describe('concat', () => {
  it('concat two vectors along axis 0', () => {
    const a = new TensorType([3], ScalarType.F32);
    const b = new TensorType([2], ScalarType.F32);
    const c = new TensorType([5], ScalarType.F32);
    const func = buildFunction('cat', [a, b], [c], (bb, args) => {
      bb.returnOp([bb.concat([args[0], args[1]], 0).getResult(0)]);
    });

    const r = compile(func);
    const i1 = new Float32Array([1, 2, 3]);
    const i2 = new Float32Array([4, 5]);
    const out = new Float32Array(5);
    r.run('cat', i1, i2, out);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
  });

  it('concat matrices along axis 1 (column concat)', () => {
    const a = new TensorType([2, 2], ScalarType.F32);
    const b = new TensorType([2, 3], ScalarType.F32);
    const c = new TensorType([2, 5], ScalarType.F32);
    const func = buildFunction('colcat', [a, b], [c], (bb, args) => {
      bb.returnOp([bb.concat([args[0], args[1]], 1).getResult(0)]);
    });

    const r = compile(func);
    const i1 = new Float32Array([1, 2, 3, 4]);
    const i2 = new Float32Array([10, 20, 30, 40, 50, 60]);
    const out = new Float32Array(10);
    r.run('colcat', i1, i2, out);
    expect(Array.from(out)).toEqual([1, 2, 10, 20, 30, 3, 4, 40, 50, 60]);
  });
});
