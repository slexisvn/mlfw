import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { countLoops, countTempBuffers } from '../../../_utils/kernel_source.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

const t = new TensorType([4], ScalarType.F32);

describe('sigmoid', () => {
  it('σ(0) = 0.5 and σ(large) ≈ 1', () => {
    const func = buildFunction('sig', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('sig', new Float32Array([0, 1, -1, 10]), out);
    expect(out[0]).toBeCloseTo(0.5, 5);
    expect(out[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 4);
    expect(out[2]).toBeCloseTo(1 / (1 + Math.exp(1)), 4);
    expect(out[3]).toBeCloseTo(1.0, 3);
  });
});

describe('gelu', () => {
  it('gelu(0) = 0 and positive inputs amplified', () => {
    const func = buildFunction('gelu_f', [t], [t], (b, args) => {
      b.returnOp([b.gelu(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('gelu_f', new Float32Array([0, 1, -1, 2]), out);
    expect(out[0]).toBeCloseTo(0, 3);
    expect(out[1]).toBeCloseTo(0.841, 2);
    expect(out[2]).toBeCloseTo(-0.159, 1);
    expect(out[3]).toBeCloseTo(1.955, 1);
  });
});

describe('silu (swish)', () => {
  it('silu(x) = x * sigmoid(x)', () => {
    const func = buildFunction('silu_f', [t], [t], (b, args) => {
      b.returnOp([b.silu(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    const inp = [0, 1, -1, 2];
    r.run('silu_f', new Float32Array(inp), out);
    for (let i = 0; i < 4; i++) {
      const expected = inp[i] / (1 + Math.exp(-inp[i]));
      expect(out[i]).toBeCloseTo(expected, 2);
    }
  });
});

describe('codegen quality — sigmoid constant inlining', () => {
  it('sigmoid produces no temp buffers — 1.0 constant inlined as literal', () => {
    const v = new TensorType([8], ScalarType.F32);
    const func = buildFunction('sig_q', [v], [v], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const src = r.getSource('sig_q');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
  });

  it('silu produces no temp buffers — sigmoid constant inlined', () => {
    const v = new TensorType([8], ScalarType.F32);
    const func = buildFunction('silu_q', [v], [v], (b, args) => {
      b.returnOp([b.silu(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const src = r.getSource('silu_q');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
  });

  it('chained activations with sigmoid produce single fused kernel', () => {
    const v = new TensorType([16], ScalarType.F32);
    const func = buildFunction('chain_sig', [v], [v], (b, args) => {
      let x = b.tanh(args[0]).getResult(0);
      x = b.sigmoid(x).getResult(0);
      x = b.neg(x).getResult(0);
      b.returnOp([x]);
    });
    const r = compile(func);
    const src = r.getSource('chain_sig');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);

    const out = new Float32Array(16);
    r.run('chain_sig', new Float32Array(16).fill(0), out);
    expect(out[0]).toBeCloseTo(-0.5, 4);
  });
});

describe('log_softmax', () => {
  it('exp(log_softmax(x)) sums to 1', () => {
    const func = buildFunction('lsm', [t], [t], (b, args) => {
      b.returnOp([b.logSoftmax(args[0]).getResult(0)]);
    });
    const r = compile(func);
    const out = new Float32Array(4);
    r.run('lsm', new Float32Array([1, 2, 3, 4]), out);
    let sum = 0;
    for (let i = 0; i < 4; i++) {
      expect(out[i]).toBeLessThanOrEqual(0);
      sum += Math.exp(out[i]);
    }
    expect(sum).toBeCloseTo(1, 4);
  });
});
