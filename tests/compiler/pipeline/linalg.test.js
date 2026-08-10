import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { matmulOutputShape } from '../../../src/tensor/utils/shape_utils.js';

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget(), opts);
}

function rng32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function bmmRef(aShape, aData, bShape, bData) {
  const a2 = aShape.length === 1 ? [1, aShape[0]] : aShape;
  const b2 = bShape.length === 1 ? [bShape[0], 1] : bShape;
  const M = a2[a2.length - 2], K = a2[a2.length - 1], N = b2[b2.length - 1];
  const aBatch = a2.slice(0, -2), bBatch = b2.slice(0, -2);
  const nb = Math.max(aBatch.length, bBatch.length);
  const batch = [];
  for (let i = 0; i < nb; i++) {
    const av = aBatch[aBatch.length - nb + i] ?? 1;
    const bv = bBatch[bBatch.length - nb + i] ?? 1;
    batch.push(Math.max(av, bv));
  }
  const total = batch.reduce((x, y) => x * y, 1);
  const out = [];
  const offsetFor = (bShapeArr, coords) => {
    let off = 0;
    for (let d = 0; d < bShapeArr.length; d++) {
      const coord = bShapeArr[d] === 1 ? 0 : coords[nb - bShapeArr.length + d];
      off = off * bShapeArr[d] + coord;
    }
    return off;
  };
  const coords = new Array(nb).fill(0);
  for (let t = 0; t < total; t++) {
    let rem = t;
    for (let d = nb - 1; d >= 0; d--) { coords[d] = rem % batch[d]; rem = Math.floor(rem / batch[d]); }
    const aOff = offsetFor(aBatch, coords) * M * K;
    const bOff = offsetFor(bBatch, coords) * K * N;
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += aData[aOff + i * K + k] * bData[bOff + k * N + j];
        out.push(s);
      }
    }
  }
  return out;
}

describe('matmul end-to-end', () => {
  it('2x3 @ 3x2 = 2x2', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('mm', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    expect(result.succeeded).toBe(true);

    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const c = new Float32Array(4);
    result.run('mm', a, b, c);
    expect(Array.from(c)).toEqual([58, 64, 139, 154]);
  });

  it('identity matrix multiplication preserves input', () => {
    const t = new TensorType([3, 3], ScalarType.F32);
    const func = buildFunction('mm_id', [t, t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const eye = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const c = new Float32Array(9);
    result.run('mm_id', a, eye, c);
    expect(Array.from(c)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('1x4 @ 4x1 = 1x1 (dot product)', () => {
    const lhs = new TensorType([1, 4], ScalarType.F32);
    const rhs = new TensorType([4, 1], ScalarType.F32);
    const out = new TensorType([1, 1], ScalarType.F32);
    const func = buildFunction('dot_prod', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([5, 6, 7, 8]);
    const c = new Float32Array(1);
    result.run('dot_prod', a, b, c);
    expect(c[0]).toBe(70);
  });
});

describe('batched / broadcast matmul vs independent reference', () => {
  const cases = [
    [[2, 3, 4], [2, 4, 5]],
    [[1, 3, 4], [5, 4, 2]],
    [[5, 3, 4], [1, 4, 2]],
    [[2, 1, 3, 4], [2, 5, 4, 6]],
    [[3, 1, 2, 4], [5, 4, 6]],
    [[3, 4], [2, 4, 5]],
    [[2, 3, 4], [4, 5]],
    [[4], [2, 4, 5]],
    [[2, 3, 4], [4]],
    [[3, 4], [4]],
    [[4], [4, 5]],
    [[4], [4]],
  ];
  for (const [aShape, bShape] of cases) {
    it(`${JSON.stringify(aShape)} @ ${JSON.stringify(bShape)}`, () => {
      const numel = (s) => s.reduce((x, y) => x * y, 1);
      const r = rng32(7919 + aShape.length * 131 + bShape.length);
      const aData = Float32Array.from({ length: numel(aShape) }, () => -1 + 2 * r());
      const bData = Float32Array.from({ length: numel(bShape) }, () => -1 + 2 * r());
      const outShape = matmulOutputShape(aShape, bShape);
      const func = buildFunction('bmm', [new TensorType(aShape, ScalarType.F32), new TensorType(bShape, ScalarType.F32)], [new TensorType(outShape, ScalarType.F32)], (b, args) => {
        b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
      });
      const result = compile(func);
      expect(result.succeeded).toBe(true);
      const out = new Float32Array(Math.max(numel(outShape), 1));
      result.run('bmm', aData, bData, out);
      const ref = bmmRef(aShape, aData, bShape, bData);
      expect(out.length).toBe(ref.length);
      for (let i = 0; i < ref.length; i++) {
        expect(Math.abs(out[i] - ref[i])).toBeLessThan(1e-3);
      }
    });
  }
});

describe('fuzz: random N-D broadcast matmul vs independent reference', () => {
  function genShapes(r) {
    const ri = (lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
    const M = ri(1, 5), K = ri(1, 5), N = ri(1, 5);
    const nBatch = ri(0, 3);
    const aBatch = [], bBatch = [];
    for (let i = 0; i < nBatch; i++) {
      const d = ri(1, 4);
      const mode = ri(0, 2);
      aBatch.push(mode === 1 ? 1 : d);
      bBatch.push(mode === 2 ? 1 : d);
    }
    let aShape = [...aBatch, M, K];
    let bShape = [...bBatch, K, N];
    const drop = ri(0, 3);
    if (drop === 1 && aShape.length === 2) aShape = [K];
    else if (drop === 2 && bShape.length === 2) bShape = [K];
    return [aShape, bShape];
  }

  it('120 random programs match the reference', () => {
    const numel = (s) => s.reduce((x, y) => x * y, 1);
    const fails = [];
    for (let s = 0; s < 120; s++) {
      const r = rng32(424242 + s * 2654435761);
      const [aShape, bShape] = genShapes(r);
      const aData = Float32Array.from({ length: numel(aShape) }, () => -1 + 2 * r());
      const bData = Float32Array.from({ length: numel(bShape) }, () => -1 + 2 * r());
      const outShape = matmulOutputShape(aShape, bShape);
      const func = buildFunction('fz', [new TensorType(aShape, ScalarType.F32), new TensorType(bShape, ScalarType.F32)], [new TensorType(outShape, ScalarType.F32)], (b, args) => {
        b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
      });
      const result = compile(func);
      if (!result.succeeded) { fails.push(`s=${s} ${JSON.stringify(aShape)}@${JSON.stringify(bShape)} compile failed`); continue; }
      const out = new Float32Array(Math.max(numel(outShape), 1));
      result.run('fz', aData, bData, out);
      const ref = bmmRef(aShape, aData, bShape, bData);
      if (out.length !== ref.length) { fails.push(`s=${s} len ${out.length}!=${ref.length}`); continue; }
      for (let i = 0; i < ref.length; i++) {
        if (Math.abs(out[i] - ref[i]) > 1e-3) { fails.push(`s=${s} ${JSON.stringify(aShape)}@${JSON.stringify(bShape)} idx${i} ${out[i]} vs ${ref[i]}`); break; }
      }
    }
    expect(fails, fails.slice(0, 8).join('\n')).toEqual([]);
  });
});

describe('matmul + elementwise chain', () => {
  it('matmul then add bias', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const bias = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('mm_add', [lhs, rhs, bias], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.add(mm.getResult(0), args[2]).getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const biasData = new Float32Array([100, 200, 300, 400]);
    const c = new Float32Array(4);
    result.run('mm_add', a, b, biasData, c);
    expect(Array.from(c)).toEqual([158, 264, 439, 554]);
  });

  it('matmul then neg', () => {
    const lhs = new TensorType([2, 2], ScalarType.F32);
    const rhs = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('mm_neg', [lhs, rhs], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 0, 0, 1]);
    const b = new Float32Array([5, 6, 7, 8]);
    const c = new Float32Array(4);
    result.run('mm_neg', a, b, c);
    expect(Array.from(c)).toEqual([-5, -6, -7, -8]);
  });
});

describe('reduce end-to-end', () => {
  it('sum reduction along axis 1 (row sum)', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const out = new TensorType([2], ScalarType.F32);

    const func = buildFunction('row_sum', [t], [out], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      b.returnOp([b.reduce(args[0], zero.getResult(0), [1], 'sum').getResult(0)]);
    });

    const result = compile(func);
    expect(result.succeeded).toBe(true);

    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const o = new Float32Array(2);
    result.run('row_sum', a, o);
    expect(Array.from(o)).toEqual([6, 15]);
  });

  it('max reduction along axis 0 (column max)', () => {
    const t = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2], ScalarType.F32);

    const func = buildFunction('col_max', [t], [out], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], negInf.getResult(0), [0], 'max').getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 6, 5, 2, 3, 4]);
    const o = new Float32Array(2);
    result.run('col_max', a, o);
    expect(Array.from(o)).toEqual([5, 6]);
  });
});

describe('codegen quality — extent-1 loop elimination', () => {
  it('batch=1 matmul emits no loop with extent 1', () => {
    const lhs = new TensorType([1, 4], ScalarType.F32);
    const rhs = new TensorType([4, 3], ScalarType.F32);
    const out = new TensorType([1, 3], ScalarType.F32);
    const func = buildFunction('mm_b1', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('mm_b1');
    const loops = src.match(/for\s*\(\s*let\s+\w+\s*=\s*0;\s*\w+\s*<\s*1;/g);
    expect(loops).toBeNull();
  });

  it('batch=1 matmul still computes correct values', () => {
    const lhs = new TensorType([1, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([1, 2], ScalarType.F32);
    const func = buildFunction('mm_b1_v', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 2, 3]);
    const b2 = new Float32Array([4, 5, 6, 7, 8, 9]);
    const c = new Float32Array(2);
    result.run('mm_b1_v', a, b2, c);
    expect(Array.from(c)).toEqual([40, 46]);
  });
});

describe('codegen quality — zero-stride index cleanup', () => {
  it('batch=1 matmul source has no 0 * stride terms', () => {
    const lhs = new TensorType([1, 4], ScalarType.F32);
    const rhs = new TensorType([4, 3], ScalarType.F32);
    const out = new TensorType([1, 3], ScalarType.F32);
    const func = buildFunction('mm_idx', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('mm_idx');
    expect(src).not.toMatch(/0\s*\*\s*\d+/);
    expect(src).not.toMatch(/\d+\s*\*\s*0/);
  });
});

describe('codegen quality — redundant zero-init elimination', () => {
  it('matmul init loop is eliminated — Float32Array is pre-zeroed', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 4], ScalarType.F32);
    const out = new TensorType([2, 4], ScalarType.F32);
    const func = buildFunction('mm_noinit', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('mm_noinit');
    const zeroStores = src.match(/\w+\[\w+\]\s*=\s*0\s*;/g) || [];
    expect(zeroStores.length).toBe(0);
  });

  it('matmul without zero-init still computes correct values', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);
    const func = buildFunction('mm_noinit_v', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const result = compile(func);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b2 = new Float32Array([7, 8, 9, 10, 11, 12]);
    const c = new Float32Array(4);
    result.run('mm_noinit_v', a, b2, c);
    expect(Array.from(c)).toEqual([58, 64, 139, 154]);
  });

});

describe('codegen quality — zero-init elimination for intermediate buffers', () => {
  it('matmul chain emits no zero-fill for intermediate accumulation buffer', () => {
    const a = new TensorType([2, 4], ScalarType.F32);
    const b = new TensorType([4, 3], ScalarType.F32);
    const c = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('mm_chain_init', [a, b, c], [out], (bb, args) => {
      const h = bb.matmul(args[0], args[1]).getResult(0);
      bb.returnOp([bb.matmul(h, args[2]).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('mm_chain_init');
    const zeroStores = src.match(/\w+\[\w+\]\s*=\s*0\s*;/g) || [];
    expect(zeroStores.length).toBe(0);
  });

  it('matmul chain without zero-init produces correct values', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([3, 3], ScalarType.F32);
    const out = new TensorType([2, 3], ScalarType.F32);

    const func = buildFunction('mm_chain_v', [a, b], [out], (bb, args) => {
      const h = bb.matmul(args[0], args[1]).getResult(0);
      bb.returnOp([bb.matmul(h, args[1]).getResult(0)]);
    });

    const result = compile(func);
    const eye = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const x = new Float32Array([1, 2, 3, 4, 5, 6]);
    const c = new Float32Array(6);
    result.run('mm_chain_v', x, eye, c);
    expect(Array.from(c)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('codegen quality — fold transpose into dot', () => {
  it('dot(x, transpose(y)) produces no transpose op at runtime', () => {
    const lhs = new TensorType([3, 5], ScalarType.F32);
    const rhs = new TensorType([7, 5], ScalarType.F32);
    const out = new TensorType([3, 7], ScalarType.F32);
    const func = buildFunction('dot_notrans', [lhs, rhs], [out], (b, args) => {
      const t = b.transpose(args[1], [1, 0]);
      b.returnOp([b.dot(args[0], t.getResult(0), [1], [0]).getResult(0)]);
    });

    const result = compile(func);
    const src = result.getSource('dot_notrans');
    expect(src).not.toMatch(/transpose/i);
  });

  it('dot(x, transpose(y)) computes same result as direct dot', () => {
    const lhs = new TensorType([2, 3], ScalarType.F32);
    const rhs = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const funcDirect = buildFunction('dot_direct', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.dot(args[0], args[1], [1], [0]).getResult(0)]);
    });
    const resDirect = compile(funcDirect);
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const bDirect = new Float32Array([7, 8, 9, 10, 11, 12]);
    const cDirect = new Float32Array(4);
    resDirect.run('dot_direct', a, bDirect, cDirect);

    const rhsT = new TensorType([2, 3], ScalarType.F32);
    const funcFolded = buildFunction('dot_folded', [lhs, rhsT], [out], (b, args) => {
      const t = b.transpose(args[1], [1, 0]);
      b.returnOp([b.dot(args[0], t.getResult(0), [1], [0]).getResult(0)]);
    });
    const resFolded = compile(funcFolded);
    const bTransposed = new Float32Array([7, 9, 11, 8, 10, 12]);
    const cFolded = new Float32Array(4);
    resFolded.run('dot_folded', a, bTransposed, cFolded);

    expect(Array.from(cFolded)).toEqual(Array.from(cDirect));
  });
});

describe('conv end-to-end', () => {
  it('1x1x4x4 conv with 1x1x2x2 kernel stride=1 no-pad', () => {
    const inp = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const ker = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const out = new TensorType([1, 1, 3, 3], ScalarType.F32);

    const func = buildFunction('conv_simple', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const result = compile(func);
    expect(result.succeeded).toBe(true);

    const input = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const kernel = new Float32Array([1, 0, 0, 1]);
    const output = new Float32Array(9);
    result.run('conv_simple', input, kernel, output);

    expect(output[0]).toBe(1 + 6);
    expect(output[1]).toBe(2 + 7);
    expect(output[2]).toBe(3 + 8);
    expect(output[3]).toBe(5 + 10);
    expect(output[8]).toBe(11 + 16);
  });
});
