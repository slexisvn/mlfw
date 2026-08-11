import { describe, it, expect } from 'vitest';
import { ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { QuantizationScheme, QuantizationParams } from '../../../../src/compiler/ir/graph/quantization_types.js';

describe('QuantizationParams.fromRange symmetric', () => {
  it('symmetric I8: scale = absMax / 127, zeroPoint = 0', () => {
    const qp = QuantizationParams.fromRange(-3, 3, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(qp.getScalarScale()).toBeCloseTo(3 / 127, 10);
    expect(qp.getScalarZeroPoint()).toBe(0);
    expect(qp.dtype).toBe(ScalarType.I8);
  });

  it('symmetric uses max(|min|, |max|) for scale', () => {
    const qp = QuantizationParams.fromRange(-1, 5, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(qp.getScalarScale()).toBeCloseTo(5 / 127, 10);
  });

  it('clampRange for symmetric I8 is [-127, 127]', () => {
    const qp = QuantizationParams.fromRange(-6, 6, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const [cMin, cMax] = qp.clampRange();
    expect(cMin).toBe(-127);
    expect(cMax).toBe(127);
  });
});

describe('QuantizationParams.fromRange asymmetric', () => {
  it('asymmetric I8: scale = range / 255, zeroPoint clamps to [-128, 127]', () => {
    const qp = QuantizationParams.fromRange(0, 6, QuantizationScheme.PER_TENSOR_ASYMMETRIC);
    expect(qp.getScalarScale()).toBeCloseTo(6 / 255, 10);
    const zp = qp.getScalarZeroPoint();
    expect(zp).toBeGreaterThanOrEqual(-128);
    expect(zp).toBeLessThanOrEqual(127);
  });

  it('asymmetric UI8: range = [0, 255]', () => {
    const qp = QuantizationParams.fromRange(0, 1, QuantizationScheme.PER_TENSOR_ASYMMETRIC, ScalarType.UI8);
    const [cMin, cMax] = qp.clampRange();
    expect(cMin).toBe(0);
    expect(cMax).toBe(255);
  });
});

describe('QuantizationParams quantize/dequantize roundtrip', () => {
  it('quantize then dequantize approximates original value', () => {
    const qp = QuantizationParams.fromRange(-6, 6, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const original = 3.5;
    const quantized = qp.quantize(original);
    const restored = qp.dequantize(quantized);
    expect(Math.abs(restored - original)).toBeLessThan(qp.getScalarScale());
  });

  it('quantize clamps to valid range', () => {
    const qp = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const [cMin, cMax] = qp.clampRange();
    expect(qp.quantize(1000)).toBe(cMax);
    expect(qp.quantize(-1000)).toBe(cMin);
  });

  it('quantizeArray / dequantizeArray roundtrip', () => {
    const qp = QuantizationParams.fromRange(-10, 10, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const original = [1.5, -3.2, 0.0, 7.8];
    const quantized = qp.quantizeArray(original);
    const restored = qp.dequantizeArray(quantized);

    for (let i = 0; i < original.length; i++) {
      expect(Math.abs(restored[i] - original[i])).toBeLessThan(qp.getScalarScale());
    }
  });

  it('zero quantizes to zero_point in symmetric scheme', () => {
    const qp = QuantizationParams.fromRange(-5, 5, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(qp.quantize(0)).toBe(0);
  });
});

describe('QuantizationParams.fromConstantArray', () => {
  it('derives range from array min/max', () => {
    const data = [1.0, -2.0, 3.0, -0.5];
    const qp = QuantizationParams.fromConstantArray(data, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(qp.getScalarScale()).toBeCloseTo(3 / 127, 10);
  });

  it('handles all-same-value array by expanding range by +-0.5', () => {
    const data = [5.0, 5.0, 5.0];
    const qp = QuantizationParams.fromConstantArray(data, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(qp.getScalarScale()).toBeCloseTo(5.5 / 127, 10);
  });
});

describe('QuantizationParams.fromRangePerChannel', () => {
  it('creates per-channel scales with correct number of channels', () => {
    const mins = [-1, -2, -3];
    const maxs = [1, 2, 3];
    const qp = QuantizationParams.fromRangePerChannel(mins, maxs, 0);

    expect(qp.isPerChannel()).toBe(true);
    expect(qp.numChannels()).toBe(3);
    expect(qp.axis).toBe(0);
    expect(qp.getScaleForChannel(0)).toBeCloseTo(1 / 127, 10);
    expect(qp.getScaleForChannel(1)).toBeCloseTo(2 / 127, 10);
    expect(qp.getScaleForChannel(2)).toBeCloseTo(3 / 127, 10);
  });

  it('per-channel zero points are all 0 (symmetric)', () => {
    const mins = [-1, -2];
    const maxs = [1, 2];
    const qp = QuantizationParams.fromRangePerChannel(mins, maxs, 1);
    expect(qp.getZeroPointForChannel(0)).toBe(0);
    expect(qp.getZeroPointForChannel(1)).toBe(0);
  });
});

describe('QuantizationParams.defaultForActivation', () => {
  it('uses range [-6, 6] for activation defaults', () => {
    const qp = QuantizationParams.defaultForActivation(QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(qp.getScalarScale()).toBeCloseTo(6 / 127, 10);
    expect(qp.getScalarZeroPoint()).toBe(0);
  });
});

describe('QuantizationParams equality and serialization', () => {
  it('equals returns true for identical params', () => {
    const a = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const b = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(a.equals(b)).toBe(true);
  });

  it('equals returns false for different scales', () => {
    const a = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const b = QuantizationParams.fromRange(-5, 5, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(a.equals(b)).toBe(false);
  });

  it('serialize/deserialize roundtrip preserves params', () => {
    const original = QuantizationParams.fromRange(-3, 3, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const restored = QuantizationParams.deserialize(original.serialize());
    expect(restored.equals(original)).toBe(true);
  });

  it('hash is deterministic and differs for different params', () => {
    const a = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const b = QuantizationParams.fromRange(-5, 5, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    expect(a.hash()).toBe(a.hash());
    expect(a.hash()).not.toBe(b.hash());
  });
});

describe('QuantizationParams per-channel from a [K=4, N=3] weight whose column n has magnitude ~(n+1), giving wildly different per-channel ranges', () => {
  const K = 4, N = 3;
  const W = [];
  for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) W.push(Math.sin(k * 1.7 + n) * (n + 1) * 3);

  it('computes one symmetric scale per output channel (axis=1)', () => {
    const qp = QuantizationParams.fromConstantArrayPerChannel(W, [K, N], 1, ScalarType.I8);
    expect(qp.isPerChannel()).toBe(true);
    expect(qp.axis).toBe(1);
    expect(qp.numChannels()).toBe(N);
    for (let n = 0; n < N; n++) {
      let absMax = 0;
      for (let k = 0; k < K; k++) absMax = Math.max(absMax, Math.abs(W[k * N + n]));
      expect(qp.getScaleForChannel(n)).toBeCloseTo(absMax / 127, 10);
      expect(qp.getZeroPointForChannel(n)).toBe(0);
    }
  });

  it('quantize/dequantize round-trip stays within the per-channel step', () => {
    const qp = QuantizationParams.fromConstantArrayPerChannel(W, [K, N], 1, ScalarType.I8);
    const q = qp.quantizeArrayPerChannel(W, [K, N]);
    const d = qp.dequantizeArrayPerChannel(q, [K, N]);
    for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) {
      const i = k * N + n;
      expect(Math.abs(d[i] - W[i])).toBeLessThanOrEqual(qp.getScaleForChannel(n) / 2 + 1e-9);
    }
  });

  it('per-channel along a non-last axis (axis=0) maps the stride correctly', () => {
    const qp = QuantizationParams.fromConstantArrayPerChannel(W, [K, N], 0, ScalarType.I8);
    expect(qp.numChannels()).toBe(K);
    for (let k = 0; k < K; k++) {
      let absMax = 0;
      for (let n = 0; n < N; n++) absMax = Math.max(absMax, Math.abs(W[k * N + n]));
      expect(qp.getScaleForChannel(k)).toBeCloseTo(absMax / 127, 10);
    }
  });

  it('accuracy harness: per-channel error is markedly lower than per-tensor on skewed weights', () => {
    const pt = QuantizationParams.fromConstantArray(W, QuantizationScheme.PER_TENSOR_SYMMETRIC, ScalarType.I8);
    const ptRec = pt.dequantizeArray(pt.quantizeArray(W));
    let ptErr = 0;
    for (let i = 0; i < W.length; i++) ptErr += Math.abs(ptRec[i] - W[i]);
    ptErr /= W.length;

    const pc = QuantizationParams.fromConstantArrayPerChannel(W, [K, N], 1, ScalarType.I8);
    const pcRec = pc.dequantizeArrayPerChannel(pc.quantizeArrayPerChannel(W, [K, N]), [K, N]);
    let pcErr = 0;
    for (let i = 0; i < W.length; i++) pcErr += Math.abs(pcRec[i] - W[i]);
    pcErr /= W.length;

    expect(pcErr).toBeLessThan(ptErr);
    expect(ptErr / pcErr).toBeGreaterThan(1.5);
  });
});

describe('INT4 group-wise quantization (GPTQ/AWQ-style)', () => {
  const N = 64, G = 16;
  const data = Array.from({ length: N }, (_, i) => Math.sin(i * 0.9) * (1 + (i % G) * 0.1));

  it('produces one scale per group at 4 bits with values in the INT4 range', () => {
    const qp = QuantizationParams.fromConstantArrayPerGroup(data, G, ScalarType.I8, 4);
    expect(qp.isPerGroup()).toBe(true);
    expect(qp.numBits).toBe(4);
    expect(qp.scale.length).toBe(Math.ceil(N / G));
    const q = qp.quantizeArrayPerGroup(data);
    expect(q.every(v => Number.isInteger(v) && v >= -7 && v <= 7)).toBe(true);
  });

  it('round-trips within the per-group quantization step', () => {
    const qp = QuantizationParams.fromConstantArrayPerGroup(data, G, ScalarType.I8, 4);
    const deq = qp.dequantizeArrayPerGroup(qp.quantizeArrayPerGroup(data));
    let e = 0;
    for (let i = 0; i < N; i++) e = Math.max(e, Math.abs(deq[i] - data[i]));
    expect(e).toBeLessThanOrEqual(Math.max(...qp.scale) / 2 + 1e-9);
  });

  it('beats per-tensor INT4 accuracy and serializes', () => {
    const grp = QuantizationParams.fromConstantArrayPerGroup(data, G, ScalarType.I8, 4);
    const pt = QuantizationParams.fromConstantArray(data, QuantizationScheme.PER_TENSOR_SYMMETRIC, ScalarType.I8, 4);
    const err = (deq) => { let e = 0; for (let i = 0; i < N; i++) e = Math.max(e, Math.abs(deq[i] - data[i])); return e; };
    const grpErr = err(grp.dequantizeArrayPerGroup(grp.quantizeArrayPerGroup(data)));
    const ptErr = err(pt.dequantizeArray(pt.quantizeArray(data)));
    expect(grpErr).toBeLessThan(ptErr);
    expect(QuantizationParams.deserialize(grp.serialize()).equals(grp)).toBe(true);
  });
});
