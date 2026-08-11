import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

const t = new TensorType([4], ScalarType.F32);

describe('abs', () => {
  it('absolute value of mixed signs', () => {
    const func = buildFunction('abs_f', [t], [t], (b, args) => {
      b.returnOp([b.abs(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('abs_f', new Float32Array([-3, 0, 2, -7]), out);
    expect(Array.from(out)).toEqual([3, 0, 2, 7]);
  });
});

describe('floor / ceil', () => {
  it('floor rounds toward negative infinity', () => {
    const func = buildFunction('floor_f', [t], [t], (b, args) => {
      b.returnOp([b.floor(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('floor_f', new Float32Array([1.7, -1.2, 0.5, 3.0]), out);
    expect(Array.from(out)).toEqual([1, -2, 0, 3]);
  });

  it('ceil rounds toward positive infinity', () => {
    const func = buildFunction('ceil_f', [t], [t], (b, args) => {
      b.returnOp([b.ceil(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('ceil_f', new Float32Array([1.7, -1.2, 0.5, 3.0]), out);
    expect(Array.from(out)).toEqual([2, -1, 1, 3]);
  });
});

describe('sign', () => {
  it('returns -1, 0, or 1', () => {
    const func = buildFunction('sign_f', [t], [t], (b, args) => {
      b.returnOp([b.sign(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('sign_f', new Float32Array([-5, 0, 3, -0.1]), out);
    expect(Array.from(out)).toEqual([-1, 0, 1, -1]);
  });
});

describe('sqrt / rsqrt', () => {
  it('sqrt of perfect squares', () => {
    const func = buildFunction('sqrt_f', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('sqrt_f', new Float32Array([4, 9, 16, 1]), out);
    expect(Array.from(out)).toEqual([2, 3, 4, 1]);
  });

  it('rsqrt = 1/sqrt', () => {
    const func = buildFunction('rsqrt_f', [t], [t], (b, args) => {
      b.returnOp([b.rsqrt(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('rsqrt_f', new Float32Array([4, 1, 0.25, 100]), out);
    expect(out[0]).toBeCloseTo(0.5, 5);
    expect(out[1]).toBeCloseTo(1, 5);
    expect(out[2]).toBeCloseTo(2, 5);
    expect(out[3]).toBeCloseTo(0.1, 5);
  });
});

describe('log', () => {
  it('natural logarithm', () => {
    const func = buildFunction('log_f', [t], [t], (b, args) => {
      b.returnOp([b.log(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('log_f', new Float32Array([1, Math.E, Math.E * Math.E, 0.5]), out);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(1, 5);
    expect(out[2]).toBeCloseTo(2, 4);
    expect(out[3]).toBeCloseTo(Math.log(0.5), 5);
  });
});

describe('sin / cos', () => {
  it('sin at key angles', () => {
    const func = buildFunction('sin_f', [t], [t], (b, args) => {
      b.returnOp([b.sin(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('sin_f', new Float32Array([0, Math.PI / 2, Math.PI, 3 * Math.PI / 2]), out);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(1, 5);
    expect(out[2]).toBeCloseTo(0, 5);
    expect(out[3]).toBeCloseTo(-1, 5);
  });

  it('cos at key angles', () => {
    const func = buildFunction('cos_f', [t], [t], (b, args) => {
      b.returnOp([b.cos(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('cos_f', new Float32Array([0, Math.PI / 2, Math.PI, 3 * Math.PI / 2]), out);
    expect(out[0]).toBeCloseTo(1, 5);
    expect(out[1]).toBeCloseTo(0, 5);
    expect(out[2]).toBeCloseTo(-1, 5);
    expect(out[3]).toBeCloseTo(0, 5);
  });
});

describe('pow', () => {
  it('base^exponent element-wise', () => {
    const func = buildFunction('pow_f', [t, t], [t], (b, args) => {
      b.returnOp([b.pow(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('pow_f', new Float32Array([2, 3, 4, 5]), new Float32Array([3, 2, 0.5, 0]), out);
    expect(out[0]).toBe(8);
    expect(out[1]).toBe(9);
    expect(out[2]).toBe(2);
    expect(out[3]).toBe(1);
  });
});

describe('rem', () => {
  it('remainder element-wise', () => {
    const func = buildFunction('rem_f', [t, t], [t], (b, args) => {
      b.returnOp([b.rem(args[0], args[1]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('rem_f', new Float32Array([7, 10, 5.5, 9]), new Float32Array([3, 4, 2, 3]), out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(2);
    expect(out[2]).toBeCloseTo(1.5, 5);
    expect(out[3]).toBe(0);
  });
});

describe('convert dtype', () => {
  it('f32 → i32 truncates toward zero', () => {
    const func = buildFunction('cvt', [t], [new TensorType([4], ScalarType.I32)], (b, args) => {
      b.returnOp([b.convert(args[0], ScalarType.I32).getResult(0)]);
    });
    const r = compile(func);
    const out = new Int32Array(4);
    r.run('cvt', new Float32Array([1.9, -2.5, 0.1, 3.7]), out);
    expect(Array.from(out)).toEqual([1, -2, 0, 3]);
  });
});
