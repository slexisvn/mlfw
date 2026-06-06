import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { QuantizationPass, QuantizationConfig } from '../../../src/compiler/passes/quantization/quantization_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { QuantizationScheme } from '../../../src/compiler/ir/graph/quantization_types.js';

function run(func, opts = {}) {
  return new QuantizationPass(opts).run(func);
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

describe('QuantizationPass target gate', () => {
  it('returns UNCHANGED when target.supportsInt8 is false', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, { target: { supportsInt8: false } })).toBe(PassResult.UNCHANGED);
  });

  it('proceeds when target.supportsInt8 is true', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, { target: { supportsInt8: true } })).toBe(PassResult.CHANGED);
  });

  it('no target config proceeds normally', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
  });
});

describe('QuantizationPass exclusion logic', () => {
  it('excludeOps skips exp (in default set)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
    expect(findOps(func, 'quantize').length).toBe(0);
  });

  it('non-quantizable ops (neg) are skipped', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('custom excludeOps overrides default set', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, { excludeOps: new Set(['dot']) })).toBe(PassResult.UNCHANGED);
  });
});

describe('QuantizationPass native dot variant', () => {
  it('replaces dot with quantized_dot and erases original', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'quantized_dot').length).toBe(1);
    expect(findOps(func, 'dot').length).toBe(0);
  });

  it('quantized_dot result type is I32', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const qDot = findOps(func, 'quantized_dot')[0];
    expect(qDot.getResult(0).type.dtype).toBe(ScalarType.I32);
  });

  it('quantized_dot inputs are quantized to I8', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const qDot = findOps(func, 'quantized_dot')[0];
    for (let i = 0; i < qDot.numOperands; i++) {
      const src = qDot.getOperand(i).definingOp;
      expect(src.opName).toBe('quantize');
      expect(src.getResult(0).type.dtype).toBe(ScalarType.I8);
    }
  });

  it('quantized_dot carries lhs/rhs scale attrs', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const qDot = findOps(func, 'quantized_dot')[0];
    expect(qDot.getAttr('lhs_scale')).toBeGreaterThan(0);
    expect(qDot.getAttr('rhs_scale')).toBeGreaterThan(0);
    expect(qDot.getAttr('lhs_zero_point')).toBe(0);
    expect(qDot.getAttr('rhs_zero_point')).toBe(0);
  });

  it('output_scale = lhs_scale * rhs_scale', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const qDot = findOps(func, 'quantized_dot')[0];
    const lhsScale = qDot.getAttr('lhs_scale');
    const rhsScale = qDot.getAttr('rhs_scale');
    const outScale = qDot.getAttr('output_scale');
    expect(outScale).toBeCloseTo(lhsScale * rhsScale, 10);
  });

  it('quantized_dot is followed by dequantize back to F32', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const ret = func.getReturnOp();
    const retSource = ret.getOperand(0).definingOp;
    expect(retSource.opName).toBe('dequantize');
    expect(retSource.getResult(0).type.dtype).toBe(ScalarType.F32);
    expect(retSource.getOperand(0).definingOp.opName).toBe('quantized_dot');
  });

  it('quantize ops preserve tensor shapes', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const quantOps = findOps(func, 'quantize');
    expect(quantOps[0].getResult(0).type.shape).toEqual([4, 8]);
    expect(quantOps[1].getResult(0).type.shape).toEqual([8, 6]);
  });

  it('dequantize after quantized_dot preserves output shape', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const dequantOps = findOps(func, 'dequantize');
    expect(dequantOps.length).toBe(1);
    expect(dequantOps[0].getResult(0).type.shape).toEqual([4, 6]);
  });

  it('default symmetric scheme uses scale = 6/127 for activation inputs', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);
    const quantOps = findOps(func, 'quantize');
    for (const q of quantOps) {
      expect(q.getAttr('scale')).toBeCloseTo(6 / 127, 10);
      expect(q.getAttr('zero_point')).toBe(0);
      expect(q.getAttr('scheme')).toBe(QuantizationScheme.PER_TENSOR_SYMMETRIC);
    }
  });
});

describe('QuantizationPass native conv variant', () => {
  it('replaces conv with quantized_conv using input/kernel naming', () => {
    const inp = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 6, 6], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    expect(findOps(func, 'quantized_conv').length).toBe(1);
    expect(findOps(func, 'conv').length).toBe(0);

    const qConv = findOps(func, 'quantized_conv')[0];
    expect(qConv.getAttr('input_scale')).toBeGreaterThan(0);
    expect(qConv.getAttr('kernel_scale')).toBeGreaterThan(0);
    expect(qConv.getAttr('output_scale')).toBeCloseTo(
      qConv.getAttr('input_scale') * qConv.getAttr('kernel_scale'), 10
    );
  });

  it('quantized_conv result is I32, followed by dequantize to F32', () => {
    const inp = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 6, 6], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    run(func);
    const qConv = findOps(func, 'quantized_conv')[0];
    expect(qConv.getResult(0).type.dtype).toBe(ScalarType.I32);

    const dequantOps = findOps(func, 'dequantize');
    expect(dequantOps.length).toBe(1);
    expect(dequantOps[0].getOperand(0).definingOp.opName).toBe('quantized_conv');
  });

  it('quantized_conv inputs are quantized to I8', () => {
    const inp = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 6, 6], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    run(func);
    const qConv = findOps(func, 'quantized_conv')[0];
    for (let i = 0; i < qConv.numOperands; i++) {
      expect(qConv.getOperand(i).definingOp.opName).toBe('quantize');
    }
  });

  it('conv strides/padding attrs are preserved on quantized_conv', () => {
    const inp = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 6, 6], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [2, 2], [1, 1, 1, 1]).getResult(0)]);
    });

    run(func);
    const qConv = findOps(func, 'quantized_conv')[0];
    expect(qConv.getAttr('strides')).toEqual([2, 2]);
    expect(qConv.getAttr('padding')).toEqual([1, 1, 1, 1]);
  });
});

describe('QuantizationPass sensitivity', () => {
  it('skips ops flagged as sensitive', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, {
      sensitivityResult: { isSensitive: () => true },
      sensitivityThreshold: 0.1
    })).toBe(PassResult.UNCHANGED);
  });

  it('quantizes ops NOT flagged as sensitive', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, {
      sensitivityResult: { isSensitive: () => false },
      sensitivityThreshold: 0.1
    })).toBe(PassResult.CHANGED);
  });

  it('sensitivityThreshold=0 disables sensitivity check', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, {
      sensitivityResult: { isSensitive: () => true },
      sensitivityThreshold: 0
    })).toBe(PassResult.CHANGED);
  });
});

describe('QuantizationPass weightOnly mode', () => {
  it('skips dot when neither operand has constant definingOp', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    expect(run(func, { weightOnly: true })).toBe(PassResult.UNCHANGED);
  });
});

describe('QuantizationPass config options', () => {
  it('targetDtype=UI8 produces UI8 quantize outputs for native variant', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func, { targetDtype: ScalarType.UI8 });

    for (const q of findOps(func, 'quantize')) {
      expect(q.getResult(0).type.dtype).toBe(ScalarType.UI8);
    }
  });

  it('PER_TENSOR_ASYMMETRIC scheme is propagated to quantize attrs', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func, { scheme: QuantizationScheme.PER_TENSOR_ASYMMETRIC });

    for (const q of findOps(func, 'quantize')) {
      expect(q.getAttr('scheme')).toBe(QuantizationScheme.PER_TENSOR_ASYMMETRIC);
    }
  });
});

describe('QuantizationPass IR structure', () => {
  it('full dot pipeline: quantize lhs, quantize rhs, quantized_dot, dequantize', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    run(func);

    const opNames = [];
    for (const op of func.ops()) opNames.push(op.opName);
    expect(opNames).toEqual(['quantize', 'quantize', 'quantized_dot', 'dequantize', 'return']);
  });

  it('full conv pipeline: quantize input, quantize kernel, quantized_conv, dequantize', () => {
    const inp = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const ker = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 6, 6], ScalarType.F32);
    const func = buildFunction('f', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [0, 0, 0, 0]).getResult(0)]);
    });

    run(func);

    const opNames = [];
    for (const op of func.ops()) opNames.push(op.opName);
    expect(opNames).toEqual(['quantize', 'quantize', 'quantized_conv', 'dequantize', 'return']);
  });

  it('two dots share no quantize ops — each gets its own pair', () => {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out, out], (b, args) => {
      const d1 = b.matmul(args[0], args[1]);
      const d2 = b.matmul(args[0], args[1]);
      b.returnOp([d1.getResult(0), d2.getResult(0)]);
    });

    run(func);

    expect(findOps(func, 'quantize').length).toBe(4);
    expect(findOps(func, 'quantized_dot').length).toBe(2);
    expect(findOps(func, 'dequantize').length).toBe(2);
  });
});
