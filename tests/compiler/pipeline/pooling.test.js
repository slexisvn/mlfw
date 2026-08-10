import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import * as ref from '../../_utils/reference_ops.js';

function compile(func) {
  return compileGraph(func, CPUTarget());
}

describe('pool2d', () => {
  const x = new TensorType([1, 1, 4, 4], ScalarType.F32);
  const y = new TensorType([1, 1, 2, 2], ScalarType.F32);

  it('max pool 2x2 stride 2', () => {
    const func = buildFunction('maxp', [x], [y], (b, args) => {
      const p = b._inferAndBuild('pool2d', [args[0]], {
        pool_type: 'max', kernel_size: [2, 2], strides: [2, 2], padding: [[0, 0], [0, 0]]
      });
      b.returnOp([p.getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const out = new Float32Array(4);
    r.run('maxp', inp, out);
    expect(Array.from(out)).toEqual([6, 8, 14, 16]);
  });

  it('avg pool 2x2 stride 2', () => {
    const func = buildFunction('avgp', [x], [y], (b, args) => {
      const p = b._inferAndBuild('pool2d', [args[0]], {
        pool_type: 'avg', kernel_size: [2, 2], strides: [2, 2], padding: [[0, 0], [0, 0]]
      });
      b.returnOp([p.getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const out = new Float32Array(4);
    r.run('avgp', inp, out);
    expect(out[0]).toBeCloseTo(3.5, 5);
    expect(out[1]).toBeCloseTo(5.5, 5);
    expect(out[2]).toBeCloseTo(11.5, 5);
    expect(out[3]).toBeCloseTo(13.5, 5);
  });

  it('pad-1 max pool matches the reference, with border windows maxing only over in-bounds elements', () => {
    const inp3 = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const out3 = new TensorType([1, 1, 5, 5], ScalarType.F32);
    const func = buildFunction('maxp_pad1', [inp3], [out3], (b, args) => {
      const p = b._inferAndBuild('pool2d', [args[0]], {
        pool_type: 'max', kernel_size: [2, 2], strides: [1, 1], padding: [[1, 1], [1, 1]]
      });
      b.returnOp([p.getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const out = new Float32Array(25);
    r.run('maxp_pad1', inp, out);

    const expected = ref.pool2d(inp, [1, 1, 4, 4], 'max', [2, 2], [1, 1], 1);
    expect(expected.shape).toEqual([1, 1, 5, 5]);
    const match = ref.expectClose(out, expected.data, 1e-6, 'maxp_pad1');
    expect(match.ok, match.message).toBe(true);
  });

  it('max pool with padding', () => {
    const inp2 = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out2 = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const func = buildFunction('maxp_pad', [inp2], [out2], (b, args) => {
      const p = b._inferAndBuild('pool2d', [args[0]], {
        pool_type: 'max', kernel_size: [2, 2], strides: [2, 2], padding: [[0, 1], [0, 1]]
      });
      b.returnOp([p.getResult(0)]);
    });
    const r = compile(func);
    const inp = new Float32Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    ]);
    const out = new Float32Array(4);
    r.run('maxp_pad', inp, out);
    expect(out[0]).toBe(5);
    expect(out[1]).toBe(6);
    expect(out[2]).toBe(8);
    expect(out[3]).toBe(9);
  });

  it('avg pool with padding divides by per-output in-bounds count when count_include_pad=false', () => {
    const inp = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const func = buildFunction('avgp_pad_excl', [inp], [out], (b, args) => {
      const p = b._inferAndBuild('pool2d', [args[0]], {
        pool_type: 'avg', kernel_size: [2, 2], strides: [2, 2],
        padding: [[0, 1], [0, 1]], count_include_pad: false
      });
      b.returnOp([p.getResult(0)]);
    });
    const r = compile(func);
    const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const output = new Float32Array(4);
    r.run('avgp_pad_excl', input, output);
    expect(output[0]).toBeCloseTo(3, 5);
    expect(output[1]).toBeCloseTo(4.5, 5);
    expect(output[2]).toBeCloseTo(7.5, 5);
    expect(output[3]).toBeCloseTo(9, 5);
  });

  it('avg pool with padding divides by full kernel size when count_include_pad=true', () => {
    const inp = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const func = buildFunction('avgp_pad_incl', [inp], [out], (b, args) => {
      const p = b._inferAndBuild('pool2d', [args[0]], {
        pool_type: 'avg', kernel_size: [2, 2], strides: [2, 2],
        padding: [[0, 1], [0, 1]], count_include_pad: true
      });
      b.returnOp([p.getResult(0)]);
    });
    const r = compile(func);
    const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const output = new Float32Array(4);
    r.run('avgp_pad_incl', input, output);
    expect(output[0]).toBeCloseTo(3, 5);
    expect(output[1]).toBeCloseTo(9 / 4, 5);
    expect(output[2]).toBeCloseTo(15 / 4, 5);
    expect(output[3]).toBeCloseTo(9 / 4, 5);
  });
});
