import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { QuantizationPass, QuantizationConfig } from '../../../../src/compiler/passes/quantization/quantization_pass.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { QuantizationScheme, QuantizationParams } from '../../../../src/compiler/ir/graph/quantization_types.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../../src/compiler/support/target.js';

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

describe('QuantizationPass target dtype bit width', () => {
  function dotScale(targetDtype) {
    const lhs = new TensorType([4, 8], ScalarType.F32);
    const rhs = new TensorType([8, 6], ScalarType.F32);
    const out = new TensorType([4, 6], ScalarType.F32);
    const func = buildFunction('f', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    run(func, { targetDtype });
    const qDot = findOps(func, 'quantized_dot')[0];
    return qDot.getAttr('lhs_scale');
  }

  it('i8 activation scale uses 8-bit symmetric bound (6/127)', () => {
    expect(dotScale(ScalarType.I8)).toBeCloseTo(6 / 127, 12);
  });

  it('i16 activation scale uses 16-bit symmetric bound (6/32767), not clobbered to 8-bit', () => {
    const scale = dotScale(ScalarType.I16);
    expect(scale).toBeCloseTo(6 / 32767, 12);
    expect(scale).toBeLessThan(6 / 127);
  });

  it('i32 activation scale uses 32-bit symmetric bound', () => {
    const bound = 2 ** 31 - 1;
    expect(dotScale(ScalarType.I32)).toBeCloseTo(6 / bound, 18);
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

describe('fake-quant (dequantize∘quantize) is LOSSY — must not be removed as identity', () => {
  const I8 = ScalarType.I8, F = ScalarType.F32;
  const qp = QuantizationParams.fromRange(-6, 6, QuantizationScheme.PER_TENSOR_SYMMETRIC, I8);
  const scale = qp.getScalarScale();
  const zp = qp.getScalarZeroPoint();
  const data = [3.5, -2.1, 5.9, 1.234, -5.9, 0.001];

  function buildFakeQuant(n) {
    return buildFunction('rt', [new TensorType([n], F)], [new TensorType([n], F)], (b, a) => {
      const q = b._buildOp('quantize', [a[0]], [new TensorType([n], I8)], { scale, zero_point: zp, scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, target_dtype: I8 });
      const dq = b._buildOp('dequantize', [q.getResult(0)], [new TensorType([n], F)], { scale, zero_point: zp, scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, target_dtype: F });
      b.returnOp([dq.getResult(0)]);
    });
  }

  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`compiled dequantize(quantize(x)) matches reference math (not identity) on ${tname}`, () => {
      const res = compileGraph(buildFakeQuant(data.length), makeTarget());
      const out = new Float32Array(data.length);
      res.run('rt', new Float32Array(data), out);
      const ref = data.map((v) => qp.dequantize(qp.quantize(v)));
      let appliedQuant = false;
      for (let i = 0; i < data.length; i++) {
        expect(Math.abs(out[i] - ref[i]), `idx ${i}: compiled=${out[i]} ref=${ref[i]}`).toBeLessThan(1e-4);
        expect(Math.abs(out[i] - data[i]), `idx ${i}: quant error must be within step`).toBeLessThanOrEqual(scale + 1e-6);
        if (Math.abs(out[i] - data[i]) > 1e-6) appliedQuant = true;
      }
      expect(appliedQuant, 'off-grid inputs produced zero rounding error, so the quantization round-trip was eliminated (identity) — fake-quant broken').toBe(true);
    });
  }
});

describe('per-channel int8 matmul (compiled) beats per-tensor on skewed weights', () => {
  const F = ScalarType.F32, I8 = ScalarType.I8, I32 = ScalarType.I32;
  const T = (s, d = F) => new TensorType(s, d);
  const M = 4, K = 8, N = 6;

  const A = new Float32Array(M * K);
  for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.7);
  const W = new Float32Array(K * N);
  for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) W[k * N + n] = Math.sin(k * 1.3 + n) * (n + 1) * 2;
  const ref = new Float32Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) { let s = 0; for (let k = 0; k < K; k++) s += A[m * K + k] * W[k * N + n]; ref[m * N + n] = s; }
  let aMax = 0; for (const v of A) aMax = Math.max(aMax, Math.abs(v)); const aScale = aMax / 127 || 1e-10;

  function buildInt8Matmul(perChannel) {
    let wInt8, scaleVec;
    if (perChannel) {
      const wp = QuantizationParams.fromConstantArrayPerChannel([...W], [K, N], 1, I8);
      wInt8 = wp.quantizeArrayPerChannel([...W], [K, N]);
      scaleVec = []; for (let n = 0; n < N; n++) scaleVec.push(aScale * wp.getScaleForChannel(n));
    } else {
      const wp = QuantizationParams.fromConstantArray([...W], QuantizationScheme.PER_TENSOR_SYMMETRIC, I8);
      wInt8 = wp.quantizeArray([...W]);
      scaleVec = []; for (let n = 0; n < N; n++) scaleVec.push(aScale * wp.getScalarScale());
    }
    return buildFunction('q', [T([M, K])], [T([M, N])], (b, a) => {
      const aq = b._buildOp('quantize', [a[0]], [T([M, K], I8)], { scale: aScale, zero_point: 0, scheme: QuantizationScheme.PER_TENSOR_SYMMETRIC, target_dtype: I8 });
      const wc = b.constant(wInt8, T([K, N], I8));
      const qd = b._buildOp('quantized_dot', [aq.getResult(0), wc.getResult(0)], [T([M, N], I32)], { lhs_contracting: [1], rhs_contracting: [0], lhs_zero_point: 0, rhs_zero_point: 0, lhs_scale: aScale, rhs_scale: 1, output_scale: 1, output_zero_point: 0 });
      const cf = b._buildOp('convert', [qd.getResult(0)], [T([M, N], F)], { target_dtype: F });
      const sv = b.constant(scaleVec, T([N], F));
      const bc = b._buildOp('broadcast_in_dim', [sv.getResult(0)], [T([M, N], F)], { broadcast_dimensions: [1], result_shape: [M, N] });
      b.returnOp([b._buildOp('mul', [cf.getResult(0), bc.getResult(0)], [T([M, N], F)], {}).getResult(0)]);
    });
  }

  function meanErr(target, perChannel) {
    const out = new Float32Array(M * N);
    compileGraph(buildInt8Matmul(perChannel), target, {}).run('q', A, out);
    let e = 0; for (let i = 0; i < out.length; i++) e += Math.abs(out[i] - ref[i]);
    return { err: e / out.length, out };
  }

  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`per-channel error < per-tensor error on ${tname}`, () => {
      const pt = meanErr(makeTarget(), false);
      const pc = meanErr(makeTarget(), true);
      expect(pc.err).toBeLessThan(pt.err);
      expect(pt.err / pc.err).toBeGreaterThan(1.5);
    });
  }

  it('per-channel compiled output is identical across cpu and wasm', () => {
    const cpu = meanErr(CPUTarget(), true).out;
    const wasm = meanErr(WasmTarget(), true).out;
    for (let i = 0; i < cpu.length; i++) expect(wasm[i]).toBeCloseTo(cpu[i], 5);
  });
});

describe('QuantizationPass auto-detects per-channel weight (constant rhs of dot)', () => {
  const F = ScalarType.F32, I8 = ScalarType.I8;
  const T = (s, d = F) => new TensorType(s, d);
  const K = 6, N = 4;
  const W = [];
  for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) W.push(Math.sin(k * 1.3 + n) * (n + 1) * (n + 1));

  function buildDot() {
    return buildFunction('mm', [T([3, K])], [T([3, N])], (b, a) => {
      const wc = b.constant([...W], T([K, N]));
      b.returnOp([b.matmul(a[0], wc.getResult(0)).getResult(0)]);
    });
  }

  it('rewrites a constant-weight matmul into quantized_dot + per-channel dequant (convert→broadcast→mul)', () => {
    const func = buildDot();
    const fn = func.functions ? func.functions().next().value : func;
    run(fn, { enabled: true, scheme: QuantizationScheme.PER_CHANNEL, quantizableOps: new Set(['dot']), target: CPUTarget() });

    const qd = findOps(fn, 'quantized_dot');
    expect(qd.length).toBe(1);
    expect(qd[0].getAttr('rhs_scale')).toBe(1);
    expect(qd[0].getResult(0).type.dtype).toBe(ScalarType.I32);
    expect(findOps(fn, 'convert').length).toBe(1);
    expect(findOps(fn, 'broadcast_in_dim').length).toBe(1);
    expect(findOps(fn, 'mul').length).toBe(1);

    const i8Const = findOps(fn, 'constant').find(c => c.getResult(0).type.dtype === I8);
    expect(i8Const).toBeTruthy();
    const scaleVec = findOps(fn, 'constant').find(c => {
      const v = c.getAttr('value');
      return v && v.length === N && c.getResult(0).type.dtype === F;
    });
    expect(scaleVec).toBeTruthy();
    const sv = Array.from(scaleVec.getAttr('value'));
    expect(new Set(sv).size).toBeGreaterThan(1);
  });

  it('compiled per-channel error beats per-tensor on skewed weights (tight activation), cpu==wasm', () => {
    const M = 4;
    const A = new Float32Array(M * K);
    for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.7) * 5.5;
    const ref = new Float32Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) { let s = 0; for (let k = 0; k < K; k++) s += A[m * K + k] * W[k * N + n]; ref[m * N + n] = s; }

    const build = () => buildFunction('q', [T([M, K])], [T([M, N])], (b, a) => {
      const wc = b.constant([...W], T([K, N]));
      b.returnOp([b.matmul(a[0], wc.getResult(0)).getResult(0)]);
    });
    const errOut = (target, scheme) => {
      const r = compileGraph(build(), target, { quantization: { enabled: true, scheme, quantizableOps: new Set(['dot']) } });
      const out = new Float32Array(M * N);
      r.run('q', A, out);
      let e = 0; for (let i = 0; i < out.length; i++) e += Math.abs(out[i] - ref[i]);
      return { err: e / out.length, out };
    };

    const pt = errOut(CPUTarget(), QuantizationScheme.PER_TENSOR_SYMMETRIC).err;
    const pc = errOut(CPUTarget(), QuantizationScheme.PER_CHANNEL);
    expect(pc.err).toBeLessThan(pt);
    expect(pt / pc.err).toBeGreaterThan(1.5);

    const wasm = errOut(WasmTarget(), QuantizationScheme.PER_CHANNEL).out;
    for (let i = 0; i < pc.out.length; i++) expect(wasm[i]).toBeCloseTo(pc.out[i], 5);
  });
});

describe('compiled elementwise quantization stays numerically correct (not raw int8 codes)', () => {
  const F = ScalarType.F32;
  const T = (s) => new TensorType(s, F);
  const SH = [8, 8];
  const n = SH[0] * SH[1];

  const make = (seed) => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = Math.sin(i * 0.31 + seed) * 1.3;
    return a;
  };
  const A = make(1), B = make(2), C = make(3), D = make(4);

  const errOf = (build, inputs, ref) => {
    const r = compileGraph(build(), CPUTarget(), { quantization: { enabled: true } });
    const out = new Float32Array(n);
    r.run('q', ...inputs, out);
    let e = 0, den = 0;
    for (let i = 0; i < n; i++) { e = Math.max(e, Math.abs(out[i] - ref[i])); den = Math.max(den, Math.abs(ref[i])); }
    return e / (1 + den);
  };

  it('add of two activations round-trips through dequantize', () => {
    const ref = new Float32Array(n);
    for (let i = 0; i < n; i++) ref[i] = A[i] + B[i];
    const build = () => buildFunction('q', [T(SH), T(SH)], [T(SH)], (b, a) => {
      b.returnOp([b.add(a[0], a[1]).getResult(0)]);
    });
    expect(errOf(build, [A, B], ref)).toBeLessThan(0.05);
  });

  it('mul of two activations round-trips through dequantize', () => {
    const ref = new Float32Array(n);
    for (let i = 0; i < n; i++) ref[i] = A[i] * B[i];
    const build = () => buildFunction('q', [T(SH), T(SH)], [T(SH)], (b, a) => {
      b.returnOp([b.mul(a[0], a[1]).getResult(0)]);
    });
    expect(errOf(build, [A, B], ref)).toBeLessThan(0.05);
  });

  it('nested add chain (quantized values feeding another op) stays correct', () => {
    const ref = new Float32Array(n);
    for (let i = 0; i < n; i++) ref[i] = (A[i] + B[i]) + (C[i] + D[i]);
    const build = () => buildFunction('q', [T(SH), T(SH), T(SH), T(SH)], [T(SH)], (b, a) => {
      const l = b.add(a[0], a[1]).getResult(0);
      const r = b.add(a[2], a[3]).getResult(0);
      b.returnOp([b.add(l, r).getResult(0)]);
    });
    expect(errOf(build, [A, B, C, D], ref)).toBeLessThan(0.05);
  });
});
