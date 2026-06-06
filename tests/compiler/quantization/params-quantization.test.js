import { describe, it, expect } from 'vitest';
import { ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { QuantizationScheme, QuantizationParams } from '../../../src/compiler/ir/graph/quantization_types.js';

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
