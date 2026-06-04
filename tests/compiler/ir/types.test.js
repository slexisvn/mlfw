import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  ScalarType, DYNAMIC, Layout, TensorType, TupleType, TokenType, FunctionType,
  typeEquals, typeToString, scalarBytes, isFloatType, isIntType, broadcastDim, promoteDtype
} from '../../../src/compiler/ir/graph/types.js';

describe('ScalarType', () => {
  it('has all expected types', () => {
    assert.equal(ScalarType.F32, 'f32');
    assert.equal(ScalarType.I64, 'i64');
    assert.equal(ScalarType.BOOL, 'bool');
    assert.equal(ScalarType.INDEX, 'index');
  });

  it('scalarBytes returns correct sizes', () => {
    assert.equal(scalarBytes(ScalarType.F16), 2);
    assert.equal(scalarBytes(ScalarType.F32), 4);
    assert.equal(scalarBytes(ScalarType.F64), 8);
    assert.equal(scalarBytes(ScalarType.I8), 1);
    assert.equal(scalarBytes(ScalarType.I32), 4);
  });

  it('type classification', () => {
    assert.ok(isFloatType(ScalarType.F32));
    assert.ok(isFloatType(ScalarType.F64));
    assert.ok(!isFloatType(ScalarType.I32));
    assert.ok(isIntType(ScalarType.I32));
    assert.ok(!isIntType(ScalarType.F32));
  });

  it('promoteDtype', () => {
    assert.equal(promoteDtype(ScalarType.F32, ScalarType.F32), ScalarType.F32);
    assert.equal(promoteDtype(ScalarType.F32, ScalarType.F64), ScalarType.F64);
    assert.equal(promoteDtype(ScalarType.I32, ScalarType.F32), ScalarType.F32);
    assert.equal(promoteDtype(ScalarType.I16, ScalarType.I32), ScalarType.I32);
  });
});

describe('Layout', () => {
  it('rowMajor identity', () => {
    const l = Layout.rowMajor(3);
    assert.deepEqual([...l.order], [0, 1, 2]);
    assert.ok(l.isIdentity());
  });

  it('columnMajor', () => {
    const l = Layout.columnMajor(3);
    assert.deepEqual([...l.order], [2, 1, 0]);
    assert.ok(!l.isIdentity());
  });

  it('equals', () => {
    assert.ok(Layout.rowMajor(3).equals(Layout.rowMajor(3)));
    assert.ok(!Layout.rowMajor(3).equals(Layout.columnMajor(3)));
  });

  it('computeStrides row-major', () => {
    const l = Layout.rowMajor(3);
    assert.deepEqual(l.computeStrides([2, 3, 4]), [12, 4, 1]);
  });

  it('computeStrides with dynamic', () => {
    const l = Layout.rowMajor(3);
    const strides = l.computeStrides([DYNAMIC, 3, 4]);
    assert.equal(strides[2], 1);
    assert.equal(strides[1], 4);
    assert.equal(strides[0], 12);
  });

  it('inverse', () => {
    const l = new Layout([2, 0, 1]);
    const inv = l.inverse();
    assert.deepEqual([...inv.order], [1, 2, 0]);
  });

  it('compose', () => {
    const a = new Layout([1, 0, 2]);
    const b = new Layout([2, 1, 0]);
    const c = a.compose(b);
    assert.deepEqual([...c.order], [2, 0, 1]);
  });

  it('hash consistency', () => {
    const a = Layout.rowMajor(3);
    const b = Layout.rowMajor(3);
    assert.equal(a.hash(), b.hash());
  });
});

describe('TensorType', () => {
  it('basic properties', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    assert.equal(t.rank, 3);
    assert.equal(t.dtype, ScalarType.F32);
    assert.ok(t.isFullyStatic);
    assert.ok(!t.hasDynamic);
    assert.equal(t.numel(), 24);
    assert.equal(t.sizeInBytes(), 96);
  });

  it('scalar tensor', () => {
    const t = new TensorType([], ScalarType.F32);
    assert.equal(t.rank, 0);
    assert.ok(t.isScalar);
    assert.equal(t.numel(), 1);
  });

  it('dynamic shape', () => {
    const t = new TensorType([DYNAMIC, 3], ScalarType.F32);
    assert.ok(!t.isFullyStatic);
    assert.ok(t.hasDynamic);
    assert.equal(t.numel(), DYNAMIC);
    assert.equal(t.sizeInBytes(), DYNAMIC);
  });

  it('equals', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([2, 3], ScalarType.F32);
    assert.ok(a.equals(b));
    assert.ok(!a.equals(new TensorType([2, 4], ScalarType.F32)));
    assert.ok(!a.equals(new TensorType([2, 3], ScalarType.F64)));
  });

  it('shapeCompatible with dynamic', () => {
    const a = new TensorType([DYNAMIC, 3], ScalarType.F32);
    const b = new TensorType([2, 3], ScalarType.F32);
    assert.ok(a.shapeCompatible(b));
    assert.ok(!a.shapeCompatible(new TensorType([2, 4], ScalarType.F32)));
  });

  it('withShape/withDtype/withLayout', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const t2 = t.withShape([4, 5]);
    assert.deepEqual([...t2.shape], [4, 5]);
    assert.equal(t2.dtype, ScalarType.F32);
    const t3 = t.withDtype(ScalarType.F64);
    assert.equal(t3.dtype, ScalarType.F64);
  });

  it('hash consistency', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([2, 3], ScalarType.F32);
    assert.equal(a.hash(), b.hash());
  });
});

describe('broadcastShape', () => {
  it('same shapes', () => {
    assert.deepEqual(TensorType.broadcastShape([2, 3], [2, 3]), [2, 3]);
  });

  it('broadcast with 1', () => {
    assert.deepEqual(TensorType.broadcastShape([2, 1], [1, 3]), [2, 3]);
  });

  it('different ranks', () => {
    assert.deepEqual(TensorType.broadcastShape([3], [2, 3]), [2, 3]);
  });

  it('scalar broadcast', () => {
    assert.deepEqual(TensorType.broadcastShape([], [2, 3]), [2, 3]);
  });

  it('incompatible', () => {
    assert.equal(TensorType.broadcastShape([2, 3], [2, 4]), null);
  });

  it('dynamic broadcast', () => {
    assert.deepEqual(TensorType.broadcastShape([DYNAMIC, 3], [2, 3]), [2, 3]);
  });

  it('dynamic with 1', () => {
    assert.deepEqual(TensorType.broadcastShape([DYNAMIC], [1]), [DYNAMIC]);
  });

  it('static with dynamic non-1', () => {
    assert.deepEqual(TensorType.broadcastShape([3], [DYNAMIC]), [3]);
  });

  it('multi-shape broadcast', () => {
    assert.deepEqual(TensorType.broadcastShape([2, 1, 4], [1, 3, 1], [2, 1, 1]), [2, 3, 4]);
  });
});

describe('broadcastDim', () => {
  it('same', () => assert.equal(broadcastDim(3, 3), 3));
  it('one is 1', () => assert.equal(broadcastDim(1, 5), 5));
  it('other is 1', () => assert.equal(broadcastDim(5, 1), 5));
  it('both 1', () => assert.equal(broadcastDim(1, 1), 1));
  it('incompatible', () => assert.equal(broadcastDim(3, 4), null));
  it('dynamic same', () => assert.equal(broadcastDim(DYNAMIC, DYNAMIC), DYNAMIC));
  it('dynamic with 1', () => assert.equal(broadcastDim(DYNAMIC, 1), DYNAMIC));
  it('static with dynamic', () => assert.equal(broadcastDim(3, DYNAMIC), 3));
});

describe('TupleType', () => {
  it('equals', () => {
    const a = new TupleType([new TensorType([2], ScalarType.F32)]);
    const b = new TupleType([new TensorType([2], ScalarType.F32)]);
    assert.ok(a.equals(b));
  });
});

describe('FunctionType', () => {
  it('equals', () => {
    const ins = [new TensorType([2], ScalarType.F32)];
    const outs = [new TensorType([3], ScalarType.F32)];
    const a = new FunctionType(ins, outs);
    const b = new FunctionType(ins, outs);
    assert.ok(a.equals(b));
  });
});

describe('typeToString', () => {
  it('tensor', () => {
    assert.equal(typeToString(new TensorType([2, 3], ScalarType.F32)), 'tensor<2x3xf32>');
  });
  it('dynamic tensor', () => {
    assert.equal(typeToString(new TensorType([DYNAMIC, 3], ScalarType.F32)), 'tensor<?x3xf32>');
  });
  it('scalar tensor', () => {
    assert.equal(typeToString(new TensorType([], ScalarType.F32)), 'tensor<xf32>');
  });
  it('token', () => {
    assert.equal(typeToString(new TokenType()), 'token');
  });
  it('tuple', () => {
    const t = new TupleType([new TensorType([2], ScalarType.F32), new TokenType()]);
    assert.equal(typeToString(t), 'tuple<tensor<2xf32>, token>');
  });
});

describe('typeEquals', () => {
  it('same type', () => {
    const t = new TensorType([2], ScalarType.F32);
    assert.ok(typeEquals(t, t));
  });
  it('equal types', () => {
    assert.ok(typeEquals(new TokenType(), new TokenType()));
  });
  it('null safety', () => {
    assert.ok(!typeEquals(null, new TokenType()));
  });
});
