import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';

function compile(func) {
  return compileGraph(func, CPUTarget());
}

describe('pad', () => {
  it('pads 2x2 matrix with zeros on all sides', () => {
    const a = new TensorType([2, 2], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const b = new TensorType([4, 4], ScalarType.F32);
    const func = buildFunction('pad_f', [a, s], [b], (bb, args) => {
      bb.returnOp([bb.pad(args[0], args[1], [1, 1], [1, 1]).getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4]);
    const pv = new Float32Array([0]);
    const out = new Float32Array(16);
    r.run('pad_f', inp, pv, out);
    expect(Array.from(out)).toEqual([
      0, 0, 0, 0,
      0, 1, 2, 0,
      0, 3, 4, 0,
      0, 0, 0, 0
    ]);
  });

  it('pads with non-zero value', () => {
    const a = new TensorType([3], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const b = new TensorType([5], ScalarType.F32);
    const func = buildFunction('pad_v', [a, s], [b], (bb, args) => {
      bb.returnOp([bb.pad(args[0], args[1], [1], [1]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(5);
    r.run('pad_v', new Float32Array([10, 20, 30]), new Float32Array([-1]), out);
    expect(Array.from(out)).toEqual([-1, 10, 20, 30, -1]);
  });
});

describe('split', () => {
  it('splits vector into unequal parts', () => {
    const a = new TensorType([6], ScalarType.F32);
    const b1 = new TensorType([2], ScalarType.F32);
    const b2 = new TensorType([4], ScalarType.F32);
    const func = buildFunction('spl', [a], [b1, b2], (bb, args) => {
      const res = bb.split(args[0], 0, [2, 4]);
      bb.returnOp([res.getResult(0), res.getResult(1)]);
    });
    const r = compile(func);
    const inp = new Float32Array([10, 20, 30, 40, 50, 60]);
    const o1 = new Float32Array(2);
    const o2 = new Float32Array(4);
    r.run('spl', inp, o1, o2);
    expect(Array.from(o1)).toEqual([10, 20]);
    expect(Array.from(o2)).toEqual([30, 40, 50, 60]);
  });
});

describe('iota', () => {
  it('generates index tensor along dimension 1', () => {
    const out = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('iota_f', [], [out], (b) => {
      b.returnOp([b.iota(1, out).getResult(0)]);
    });
    const r = compile(func);
    const o = new Float32Array(6);
    r.run('iota_f', o);
    expect(Array.from(o)).toEqual([0, 1, 2, 0, 1, 2]);
  });

  it('iota along dim 0', () => {
    const out = new TensorType([3, 2], ScalarType.F32);
    const func = buildFunction('iota0', [], [out], (b) => {
      b.returnOp([b.iota(0, out).getResult(0)]);
    });
    const r = compile(func);
    const o = new Float32Array(6);
    r.run('iota0', o);
    expect(Array.from(o)).toEqual([0, 0, 1, 1, 2, 2]);
  });
});

describe('one_hot', () => {
  it('encodes indices to one-hot vectors', () => {
    const idx = new TensorType([3], ScalarType.I32);
    const out = new TensorType([3, 4], ScalarType.F32);
    const func = buildFunction('oh', [idx], [out], (b, args) => {
      b.returnOp([b.oneHot(args[0], 4, { onValue: 1, offValue: 0, dtype: ScalarType.F32 }).getResult(0)]);
    });
    const r = compile(func);
    const o = new Float32Array(12);
    r.run('oh', new Int32Array([0, 2, 3]), o);
    expect(Array.from(o)).toEqual([
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
  });
});

describe('compare + select', () => {
  it('implements element-wise max via compare+select', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('emax', [t, t], [t], (b, args) => {
      const cmp = b.compare(args[0], args[1], 'gt');
      const sel = b.select(cmp.getResult(0), args[0], args[1]);
      b.returnOp([sel.getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('emax', new Float32Array([1, 5, 3, 8]), new Float32Array([4, 2, 7, 6]), out);
    expect(Array.from(out)).toEqual([4, 5, 7, 8]);
  });
});

describe('where', () => {
  it('selects from two tensors based on condition', () => {
    const cond = new TensorType([4], ScalarType.I32);
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('whr', [cond, t, t], [t], (b, args) => {
      b.returnOp([b.where(args[0], args[1], args[2]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('whr', new Int32Array([1, 0, 1, 0]),
      new Float32Array([10, 20, 30, 40]),
      new Float32Array([100, 200, 300, 400]), out);
    expect(Array.from(out)).toEqual([10, 200, 30, 400]);
  });
});
