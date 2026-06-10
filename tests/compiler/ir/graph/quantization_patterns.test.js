import { describe, it, expect } from 'vitest';
import { buildFunction, IRBuilder } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { QuantizationScheme } from '../../../../src/compiler/ir/graph/quantization_types.js';
import { DequantizeFoldIntoDot } from '../../../../src/compiler/ir/graph/quantization_patterns.js';

function i8(shape) { return new TensorType(shape, ScalarType.I8); }
function f32(shape) { return new TensorType(shape, ScalarType.F32); }

function buildDotOfDequants(lhsZP, rhsZP) {
  const func = buildFunction('m', [i8([2, 3]), i8([3, 2])], [f32([2, 2])], (b, args) => {
    const lhsDq = b._inferAndBuild('dequantize', [args[0]], {
      scale: 0.5, zero_point: lhsZP, scheme: QuantizationScheme.PER_TENSOR_ASYMMETRIC, target_dtype: ScalarType.F32,
    });
    const rhsDq = b._inferAndBuild('dequantize', [args[1]], {
      scale: 0.25, zero_point: rhsZP, scheme: QuantizationScheme.PER_TENSOR_ASYMMETRIC, target_dtype: ScalarType.F32,
    });
    const dot = b.dot(lhsDq.getResult(0), rhsDq.getResult(0), [1], [0]);
    b.returnOp([dot.getResult(0)]);
  });
  return func;
}

describe('DequantizeFoldIntoDot zero-point guard', () => {
  const pattern = new DequantizeFoldIntoDot();

  it('folds when both zero points are zero', () => {
    const func = buildDotOfDequants(0, 0);
    const dot = func.findOp(o => o.opName === 'dot');
    const builder = new IRBuilder(func);
    builder.setInsertionPoint(dot);
    expect(pattern.match(dot)).toBe(true);
    expect(pattern.rewrite(dot, builder)).toBe(true);
    expect(func.findOp(o => o.opName === 'quantized_dot')).not.toBeNull();
  });

  it('does NOT fold when lhs zero point is non-zero', () => {
    const func = buildDotOfDequants(5, 0);
    const dot = func.findOp(o => o.opName === 'dot');
    const builder = new IRBuilder(func);
    builder.setInsertionPoint(dot);
    expect(pattern.rewrite(dot, builder)).toBe(false);
    expect(func.findOp(o => o.opName === 'quantized_dot')).toBeNull();
  });

  it('does NOT fold when rhs zero point is non-zero', () => {
    const func = buildDotOfDequants(0, -3);
    const dot = func.findOp(o => o.opName === 'dot');
    const builder = new IRBuilder(func);
    builder.setInsertionPoint(dot);
    expect(pattern.rewrite(dot, builder)).toBe(false);
    expect(func.findOp(o => o.opName === 'quantized_dot')).toBeNull();
  });
});
