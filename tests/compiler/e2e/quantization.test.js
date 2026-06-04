import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget } from '../../../src/compiler/backend/target.js';
import { Compiler, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { QuantizationParams, QuantizationScheme } from '../../../src/compiler/ir/graph/quantization_types.js';
import { CalibrationCollector } from '../../../src/compiler/analysis/calibration.js';
import { QuantizationPass } from '../../../src/compiler/passes/quantization/quantization_pass.js';
import { PassManager } from '../../../src/compiler/passes/pass_manager.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;

describe('QuantizationParams', () => {
  it('quantizes and dequantizes with round-trip error', () => {
    const qp = QuantizationParams.fromRange(-6, 6, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const original = 3.0;
    const quantized = qp.quantize(original);
    const dequantized = qp.dequantize(quantized);
    assert.ok(Math.abs(original - dequantized) < 0.1);
  });

  it('clamps to valid range', () => {
    const qp = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const [cMin, cMax] = qp.clampRange();
    assert.ok(qp.quantize(100) <= cMax);
    assert.ok(qp.quantize(-100) >= cMin);
  });

  it('symmetric has zero_point = 0', () => {
    const qp = QuantizationParams.fromRange(-5, 5, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    assert.equal(qp.zeroPoint, 0);
    assert.ok(qp.isSymmetric());
  });

  it('asymmetric has non-zero zero_point', () => {
    const qp = QuantizationParams.fromRange(0, 6, QuantizationScheme.PER_TENSOR_ASYMMETRIC, ScalarType.UI8);
    assert.ok(qp.scale > 0);
  });

  it('per-channel creates array scales', () => {
    const qp = QuantizationParams.fromRangePerChannel(
      [-1, -2, -3], [1, 2, 3], 0
    );
    assert.ok(qp.isPerChannel());
    assert.equal(qp.numChannels(), 3);
    assert.ok(qp.getScaleForChannel(0) < qp.getScaleForChannel(2));
  });

  it('equals returns true for identical params', () => {
    const a = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const b = QuantizationParams.fromRange(-1, 1, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    assert.ok(a.equals(b));
  });

  it('serialize/deserialize roundtrip', () => {
    const original = QuantizationParams.fromRange(-5, 5, QuantizationScheme.PER_TENSOR_SYMMETRIC);
    const restored = QuantizationParams.deserialize(original.serialize());
    assert.ok(original.equals(restored));
  });
});

describe('CalibrationCollector', () => {
  it('tracks min/max per value', () => {
    const func = buildFunction('calib', [new TensorType([4], f32)], [new TensorType([4], f32)],
      (b, [x]) => {
        const e = b.exp(x);
        b.returnOp([e.getResult(0)]);
      }
    );
    const collector = new CalibrationCollector('minmax');
    collector.attach(func);
    for (const [val, obs] of collector.observers) {
      obs.update(new Float32Array([-3, -1, 0, 1, 2, 3]));
    }
    const result = collector.getResult();
    for (const val of result.values()) {
      const range = result.getRange(val);
      assert.ok(range);
      assert.equal(range.min, -3);
      assert.equal(range.max, 3);
    }
  });

  it('generates quantization params from calibration', () => {
    const func = buildFunction('qp', [new TensorType([4], f32)], [new TensorType([4], f32)],
      (b, [x]) => { b.returnOp([x]); }
    );
    const collector = new CalibrationCollector('minmax');
    collector.attach(func);
    for (const [, obs] of collector.observers) {
      obs.update(new Float32Array([-6, 6]));
    }
    const result = collector.getResult();
    for (const val of result.values()) {
      const qp = result.getQuantParams(val, QuantizationScheme.PER_TENSOR_SYMMETRIC, ScalarType.I8);
      assert.ok(qp);
      assert.ok(qp.scale > 0);
    }
  });
});

describe('QuantizationPass', () => {
  it('inserts quantize/dequantize ops with calibration', () => {
    const func = buildFunction('qpass',
      [new TensorType([4, 8], f32), new TensorType([8, 4], f32)],
      [new TensorType([4, 4], f32)],
      (b, [a, w]) => {
        const mm = b.matmul(a, w);
        b.returnOp([mm.getResult(0)]);
      }
    );
    const collector = new CalibrationCollector('minmax');
    collector.attach(func);
    for (const [, obs] of collector.observers) {
      obs.update(new Float32Array([-3, -1, 0, 1, 2, 3]));
    }
    const mod = new GraphModule('test');
    mod.addFunction(func);
    const pm = new PassManager();
    pm.addPass(new QuantizationPass({ calibration: collector.getResult() }));
    const result = pm.run(mod);
    assert.equal(result.changed, true);

    let hasQuantize = false, hasDequantize = false, hasQuantizedDot = false;
    for (const op of func.ops()) {
      if (op.opName === 'quantize') hasQuantize = true;
      if (op.opName === 'dequantize') hasDequantize = true;
      if (op.opName === 'quantized_dot') hasQuantizedDot = true;
    }
    assert.ok(hasQuantize || hasQuantizedDot);
    assert.ok(hasDequantize);
  });

  it('compiles quantized graph end-to-end', () => {
    const func = buildFunction('e2e_quant',
      [new TensorType([4, 8], f32), new TensorType([8, 4], f32)],
      [new TensorType([4, 4], f32)],
      (b, [a, w]) => {
        const mm = b.matmul(a, w);
        b.returnOp([mm.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableQuantization: true,
      quantizationConfig: { weightOnly: true },
      enableFusion: false,
    });
    assert.ok(compiled.listKernels().includes('e2e_quant'));
  });
});
