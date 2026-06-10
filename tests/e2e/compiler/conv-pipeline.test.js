import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';

function compile(func) {
  return compileGraph(func, CPUTarget());
}

describe('conv2d + bias end-to-end through compiler', () => {
  it('conv kernel=1 with bias adds per-channel offset', () => {
    const x = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const k = new TensorType([2, 1, 1, 1], ScalarType.F32);
    const bias = new TensorType([1, 2, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 2, 2, 2], ScalarType.F32);

    const func = buildFunction('conv_bias', [x, k, bias], [out], (b, args) => {
      const conv = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const added = b.add(conv, args[2]).getResult(0);
      b.returnOp([added]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4]);
    const ker = new Float32Array([1, 1]);
    const biasData = new Float32Array([100, 200]);
    const result = new Float32Array(8);
    r.run('conv_bias', inp, ker, biasData, result);

    expect(result[0]).toBeCloseTo(101);
    expect(result[1]).toBeCloseTo(102);
    expect(result[2]).toBeCloseTo(103);
    expect(result[3]).toBeCloseTo(104);
    expect(result[4]).toBeCloseTo(201);
    expect(result[5]).toBeCloseTo(202);
    expect(result[6]).toBeCloseTo(203);
    expect(result[7]).toBeCloseTo(204);
  });

  it('conv kernel=3 without bias produces correct spatial reduction', () => {
    const x = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const k = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 2, 2], ScalarType.F32);

    const func = buildFunction('conv_nobias', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array(16).fill(1);
    const ker = new Float32Array(9).fill(1);
    const result = new Float32Array(4);
    r.run('conv_nobias', inp, ker, result);

    expect(result[0]).toBeCloseTo(9);
    expect(result[1]).toBeCloseTo(9);
    expect(result[2]).toBeCloseTo(9);
    expect(result[3]).toBeCloseTo(9);
  });
});

describe('codegen quality — zero-padding bounds check elision', () => {
  it('conv with pad=0 emits no bounds checks', () => {
    const x = new TensorType([1, 2, 8, 8], ScalarType.F32);
    const k = new TensorType([2, 2, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 2, 6, 6], ScalarType.F32);

    const func = buildFunction('conv_nopad', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_nopad');
    expect(src).not.toMatch(/>=\s*0/);
    expect(src).not.toMatch(/<\s*\d+\s*\)/);
  });

  it('conv with padding still emits bounds checks', () => {
    const x = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const k = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 4, 4], ScalarType.F32);

    const func = buildFunction('conv_pad', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_pad');
    expect(src).toMatch(/>=\s*0/);
  });

  it('conv with pad=0 still computes correct values', () => {
    const x = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const k = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 2, 2], ScalarType.F32);

    const func = buildFunction('conv_nopad_v', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array(16).fill(1);
    const ker = new Float32Array(9).fill(1);
    const result = new Float32Array(4);
    r.run('conv_nopad_v', inp, ker, result);
    expect(result[0]).toBe(9);
    expect(result[3]).toBe(9);
  });
});

describe('codegen quality — 1x1 conv arithmetic simplification', () => {
  it('1x1 conv has no + 0 or * 0 terms in index expressions', () => {
    const x = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const k = new TensorType([8, 4, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 8, 8, 8], ScalarType.F32);

    const func = buildFunction('conv_1x1', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_1x1');
    expect(src).not.toMatch(/\+\s*0\b/);
    expect(src).not.toMatch(/\b0\s*\*/);
  });

  it('1x1 conv computes correct pointwise channel mixing', () => {
    const x = new TensorType([1, 2, 2, 2], ScalarType.F32);
    const k = new TensorType([1, 2, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 1, 2, 2], ScalarType.F32);

    const func = buildFunction('conv_1x1_v', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]);
    const ker = new Float32Array([1, 1]);
    const result = new Float32Array(4);
    r.run('conv_1x1_v', inp, ker, result);
    expect(Array.from(result)).toEqual([11, 22, 33, 44]);
  });
});

describe('conv + relu + pool pipeline', () => {
  it('conv -> relu -> max_pool compiles and runs without errors', () => {
    const x = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const k = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 3, 3], ScalarType.F32);

    const func = buildFunction('crp', [x, k], [out], (b, args) => {
      const conv = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const relu = b.relu(conv).getResult(0);
      const pool = b.pool2d(relu, 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0);
      b.returnOp([pool]);
    });

    const r = compile(func);
    const inp = new Float32Array(64).fill(1);
    const ker = new Float32Array(9).fill(1);
    const result = new Float32Array(9);
    r.run('crp', inp, ker, result);

    expect(result.every(v => isFinite(v))).toBe(true);
    expect(result[0]).toBeCloseTo(9);
  });
});

describe('conv 3D — distinct D/H/W kernel sizes', () => {
  it('NCDHW conv with kernel D=2 H=1 W=1 reduces only the depth axis', () => {
    const inShape = [1, 1, 2, 2, 2];
    const kShape = [1, 1, 2, 1, 1];
    const outShape = [1, 1, 1, 2, 2];
    const x = new TensorType(inShape, ScalarType.F32);
    const k = new TensorType(kShape, ScalarType.F32);
    const out = new TensorType(outShape, ScalarType.F32);

    const func = buildFunction('conv3d', [x, k], [out], (b, args) => {
      const conv = b._inferAndBuild('conv', [args[0], args[1]], {
        strides: [1, 1, 1],
        padding: [[0, 0], [0, 0], [0, 0]],
        dilation: [1, 1, 1],
        groups: 1,
        input_layout: 'NCDHW',
        kernel_layout: 'OIDHW'
      }).getResult(0);
      b.returnOp([conv]);
    });

    const r = compile(func);
    const inp = new Float32Array([
      1, 2,
      3, 4,
      10, 20,
      30, 40
    ]);
    const ker = new Float32Array([1, 1]);
    const result = new Float32Array(4);
    r.run('conv3d', inp, ker, result);

    expect(result[0]).toBeCloseTo(11);
    expect(result[1]).toBeCloseTo(22);
    expect(result[2]).toBeCloseTo(33);
    expect(result[3]).toBeCloseTo(44);
  });
});
