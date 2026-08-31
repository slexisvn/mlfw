import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { countLoops, countTempBuffers } from '../../../_utils/kernel_source.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

describe('relu', () => {
  it('zeros out negatives, passes positives', () => {
    const t = new TensorType([6], ScalarType.F32);
    const func = buildFunction('relu', [t], [t], (b, args) => {
      b.returnOp([b.relu(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('relu');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
    expect(src).toMatch(/Math\.max/);
    const inp = new Float32Array([-3, -1, 0, 1, 3, 5]);
    const out = new Float32Array(6);
    r.run('relu', inp, out);
    expect(Array.from(out)).toEqual([0, 0, 0, 1, 3, 5]);
  });
});

describe('softmax', () => {
  it('row-wise softmax sums to 1', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const func = buildFunction('sm', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 1, 1, 1]);
    const out = new Float32Array(6);
    r.run('sm', inp, out);

    const row0sum = out[0] + out[1] + out[2];
    const row1sum = out[3] + out[4] + out[5];
    expect(row0sum).toBeCloseTo(1.0, 5);
    expect(row1sum).toBeCloseTo(1.0, 5);

    expect(out[2]).toBeGreaterThan(out[1]);
    expect(out[1]).toBeGreaterThan(out[0]);

    expect(out[3]).toBeCloseTo(out[4], 5);
    expect(out[4]).toBeCloseTo(out[5], 5);
  });

  it('large values do not overflow (numerical stability)', () => {
    const t = new TensorType([3], ScalarType.F32);
    const func = buildFunction('sm_big', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1000, 1001, 1002]);
    const out = new Float32Array(3);
    r.run('sm_big', inp, out);

    const sum = out[0] + out[1] + out[2];
    expect(sum).toBeCloseTo(1.0, 4);
    expect(out[0]).toBeGreaterThan(0);
    expect(out[2]).toBeGreaterThan(out[1]);
  });
});

describe('softmax decomposition quality', () => {
  it('softmax decomposes — no softmax op remains in generated source', () => {
    const t = new TensorType([2, 4], ScalarType.F32);
    const func = buildFunction('sm_dec', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('sm_dec');
    expect(src).not.toMatch(/softmax/i);
    expect(src).toMatch(/Math\.exp/);
  });

  it('softmax axis=0 column-wise produces correct distribution', () => {
    const t = new TensorType([3, 2], ScalarType.F32);
    const func = buildFunction('sm_ax0', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 0).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 0, 2, 0, 3, 0]);
    const out = new Float32Array(6);
    r.run('sm_ax0', inp, out);

    const col0sum = out[0] + out[2] + out[4];
    const col1sum = out[1] + out[3] + out[5];
    expect(col0sum).toBeCloseTo(1.0, 5);
    expect(col1sum).toBeCloseTo(1.0, 5);
    expect(out[4]).toBeGreaterThan(out[2]);
    expect(out[2]).toBeGreaterThan(out[0]);
  });

  it('matmul → softmax chain compiles and produces valid output', () => {
    const inp = new TensorType([1, 3], ScalarType.F32);
    const w = new TensorType([3, 4], ScalarType.F32);
    const out = new TensorType([1, 4], ScalarType.F32);
    const func = buildFunction('mm_sm', [inp, w], [out], (b, args) => {
      const logits = b.matmul(args[0], args[1]);
      b.returnOp([b.softmax(logits.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const x = new Float32Array([1, 0, 0]);
    const ww = new Float32Array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
    const o = new Float32Array(4);
    r.run('mm_sm', x, ww, o);

    const sum = o[0] + o[1] + o[2] + o[3];
    expect(sum).toBeCloseTo(1.0, 5);
    expect(o[3]).toBeGreaterThan(o[0]);
  });
});

describe('tanh', () => {
  it('tanh(0)=0, tanh(large)≈1, tanh(-large)≈-1', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('th', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([0, 100, -100, 1]);
    const out = new Float32Array(4);
    r.run('th', inp, out);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(1, 5);
    expect(out[2]).toBeCloseTo(-1, 5);
    expect(out[3]).toBeCloseTo(Math.tanh(1), 5);
  });
});

describe('clamp', () => {
  it('clamps values to [lo, hi] range', () => {
    const t = new TensorType([6], ScalarType.F32);
    const func = buildFunction('clmp', [t], [t], (b, args) => {
      const lo = b.broadcast(b.scalarConstant(-1, ScalarType.F32).getResult(0), [6], []);
      const hi = b.broadcast(b.scalarConstant(1, ScalarType.F32).getResult(0), [6], []);
      b.returnOp([b.clamp(lo.getResult(0), args[0], hi.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('clmp');
    expect(src).toMatch(/Math\.min/);
    expect(src).toMatch(/Math\.max/);
    const inp = new Float32Array([-5, -0.5, 0, 0.5, 5, 100]);
    const out = new Float32Array(6);
    r.run('clmp', inp, out);
    expect(Array.from(out)).toEqual([-1, -0.5, 0, 0.5, 1, 1]);
  });
});

describe('2-layer MLP: matmul → relu → matmul', () => {
  it('produces correct output for known weights', () => {
    const inp = new TensorType([2, 4], ScalarType.F32);
    const w1 = new TensorType([4, 3], ScalarType.F32);
    const w2 = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('mlp2', [inp, w1, w2], [out], (b, args) => {
      const h = b.matmul(args[0], args[1]);
      const act = b.relu(h.getResult(0));
      b.returnOp([b.matmul(act.getResult(0), args[2]).getResult(0)]);
    });

    const r = compile(func);
    const x = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]);
    const ww1 = new Float32Array([1, -1, 0, 2, 1, -1, 0, 0, 1, -1, 0, 0]);
    const ww2 = new Float32Array([1, 0, 0, 1, 1, 1]);
    const o = new Float32Array(4);
    r.run('mlp2', x, ww1, ww2, o);
    expect(Array.from(o)).toEqual([1, 0, 2, 1]);
  });
});

describe('matmul → softmax (classifier output)', () => {
  it('produces valid probability distribution', () => {
    const inp = new TensorType([2, 3], ScalarType.F32);
    const w = new TensorType([3, 4], ScalarType.F32);
    const out = new TensorType([2, 4], ScalarType.F32);

    const func = buildFunction('classify', [inp, w], [out], (b, args) => {
      const logits = b.matmul(args[0], args[1]);
      b.returnOp([b.softmax(logits.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const x = new Float32Array([1, 0, 0, 0, 0, 1]);
    const ww = new Float32Array([
      1, 2, 0, 0,
      0, 0, 1, 2,
      0, 0, 0, 0,
    ]);
    const o = new Float32Array(8);
    r.run('classify', x, ww, o);

    const row0sum = o[0] + o[1] + o[2] + o[3];
    const row1sum = o[4] + o[5] + o[6] + o[7];
    expect(row0sum).toBeCloseTo(1.0, 5);
    expect(row1sum).toBeCloseTo(1.0, 5);

    expect(o[1]).toBeGreaterThan(o[0]);
    expect(o[6] + o[7]).toBeGreaterThanOrEqual(o[4] + o[5]);
  });
});

describe('batched matmul (3D)', () => {
  it('batch dimension preserved, per-batch matmul correct', () => {
    const a = new TensorType([2, 2, 3], ScalarType.F32);
    const b = new TensorType([2, 3, 1], ScalarType.F32);
    const c = new TensorType([2, 2, 1], ScalarType.F32);

    const func = buildFunction('bmm', [a, b], [c], (bb, args) => {
      bb.returnOp([bb.matmul(args[0], args[1]).getResult(0)]);
    });

    const r = compile(func);
    const lhs = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      1, 1, 1,
      2, 2, 2,
    ]);
    const rhs = new Float32Array([
      10, 20, 30,
      1, 1, 1,
    ]);
    const out = new Float32Array(4);
    r.run('bmm', lhs, rhs, out);
    expect(Array.from(out)).toEqual([10, 20, 3, 6]);
  });
});

describe('reduce chains', () => {
  it('mean then neg (normalize-like pattern)', () => {
    const t = new TensorType([3, 4], ScalarType.F32);
    const mean_t = new TensorType([3], ScalarType.F32);

    const func = buildFunction('mean_neg', [t], [mean_t], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const m = b.reduce(args[0], zero.getResult(0), [1], 'mean');
      b.returnOp([b.neg(m.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([4, 8, 12, 16, 1, 2, 3, 4, 0, 0, 0, 0]);
    const out = new Float32Array(3);
    r.run('mean_neg', inp, out);
    expect(out[0]).toBe(-10);
    expect(out[1]).toBe(-2.5);
    expect(out[2]).toBeCloseTo(0);
  });

  it('sum over all dims (global sum)', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const step = new TensorType([2], ScalarType.F32);
    const out = new TensorType([1], ScalarType.F32);

    const func = buildFunction('gsum', [t], [out], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32);
      const partial = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      const reshp = b.reshape(partial.getResult(0), [1, 2]);
      const total = b.reduce(reshp.getResult(0), zero.getResult(0), [1], 'sum');
      b.returnOp([total.getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6]);
    const o = new Float32Array(1);
    r.run('gsum', inp, o);
    expect(o[0]).toBe(21);
  });

  it('reduce prod along axis 1', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const out = new TensorType([2], ScalarType.F32);
    const func = buildFunction('rprod', [t], [out], (b, args) => {
      const one = b.scalarConstant(1, ScalarType.F32);
      b.returnOp([b.reduce(args[0], one.getResult(0), [1], 'prod').getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([1, 2, 3, 4, 5, 6]);
    const o = new Float32Array(2);
    r.run('rprod', inp, o);
    expect(Array.from(o)).toEqual([6, 120]);
  });

  it('reduce min along axis 0', () => {
    const t = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2], ScalarType.F32);
    const func = buildFunction('cmin', [t], [out], (b, args) => {
      const inf = b.scalarConstant(Infinity, ScalarType.F32);
      b.returnOp([b.reduce(args[0], inf.getResult(0), [0], 'min').getResult(0)]);
    });

    const r = compile(func);
    const inp = new Float32Array([5, 2, 1, 8, 3, 4]);
    const o = new Float32Array(2);
    r.run('cmin', inp, o);
    expect(Array.from(o)).toEqual([1, 2]);
  });
});

describe('conv variations', () => {
  it('conv with same-padding preserves spatial dims', () => {
    const inp = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const ker = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const func = buildFunction('conv_same', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
    });

    const r = compile(func);
    const input = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    const kernel = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]);
    const output = new Float32Array(9);
    r.run('conv_same', input, kernel, output);
    expect(output[4]).toBe(1);
    expect(output[0]).toBe(1);
    expect(output[8]).toBe(1);
  });

  it('conv stride=2 downsamples spatial', () => {
    const inp = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const ker = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const out = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const func = buildFunction('conv_s2', [inp, ker], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const input = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const kernel = new Float32Array([1, 0, 0, 0]);
    const output = new Float32Array(4);
    r.run('conv_s2', input, kernel, output);
    expect(Array.from(output)).toEqual([1, 3, 9, 11]);
  });

  it('conv → relu (typical CNN layer)', () => {
    const inp = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const ker = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const out = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const func = buildFunction('conv_relu', [inp, ker], [out], (b, args) => {
      const c = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]);
      b.returnOp([b.relu(c.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_relu');
    expect(src).toMatch(/Math\.max/);
    const reluTempBuffers = (src.match(/new Float32Array/g) || []).length;
    expect(reluTempBuffers).toBeLessThanOrEqual(1);
    const input = new Float32Array([
      1, -1, 1, -1,
      -1, 1, -1, 1,
      1, -1, 1, -1,
      -1, 1, -1, 1,
    ]);
    const kernel = new Float32Array([1, 1, 1, 1]);
    const output = new Float32Array(9);
    r.run('conv_relu', input, kernel, output);
    for (let i = 0; i < 9; i++) {
      expect(output[i]).toBeGreaterThanOrEqual(0);
    }
    expect(output[0]).toBe(0);
  });
});

describe('multi-output functions', () => {
  it('returns matmul result and its transpose', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b = new TensorType([3, 2], ScalarType.F32);
    const out1 = new TensorType([2, 2], ScalarType.F32);
    const out2 = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('mm_tr', [a, b], [out1, out2], (bb, args) => {
      const mm = bb.matmul(args[0], args[1]);
      const tr = bb.transpose(mm.getResult(0), [1, 0]);
      bb.returnOp([mm.getResult(0), tr.getResult(0)]);
    });

    const r = compile(func);
    const x = new Float32Array([1, 0, 0, 0, 1, 0]);
    const y = new Float32Array([1, 2, 3, 4, 5, 6]);
    const o1 = new Float32Array(4);
    const o2 = new Float32Array(4);
    r.run('mm_tr', x, y, o1, o2);
    expect(Array.from(o1)).toEqual([1, 2, 3, 4]);
    expect(Array.from(o2)).toEqual([1, 3, 2, 4]);
  });

  it('returns the same value twice — each output gets its own buffer and copy, neither left all-zero', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('two_same', [t, t], [t, t], (b, args) => {
      const s = b.add(args[0], args[1]);
      b.returnOp([s.getResult(0), s.getResult(0)]);
    });
    const r = compile(func);
    const o1 = new Float32Array(4);
    const o2 = new Float32Array(4);
    r.run('two_same', new Float32Array([1, 2, 3, 4]), new Float32Array([10, 20, 30, 40]), o1, o2);
    expect(Array.from(o1)).toEqual([11, 22, 33, 44]);
    expect(Array.from(o2)).toEqual([11, 22, 33, 44]);
  });
});

describe('fusion data dependency ordering', () => {
  it('fused ops scheduled after all input-producing ops (SwiGLU pattern)', () => {
    const x = new TensorType([2, 4], ScalarType.F32);
    const w1 = new TensorType([4, 4], ScalarType.F32);
    const w2 = new TensorType([4, 4], ScalarType.F32);
    const out = new TensorType([2, 4], ScalarType.F32);

    const func = buildFunction('dep_order', [x, w1, w2], [out], (b, args) => {
      const gate = b.matmul(args[0], args[1]).getResult(0);
      const gateAct = b.silu(gate).getResult(0);
      const up = b.matmul(args[0], args[2]).getResult(0);
      const gated = b.mul(gateAct, up).getResult(0);
      b.returnOp([gated]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array(8).fill(1);
    const w1Data = new Float32Array(16).fill(0.5);
    const w2Data = new Float32Array(16).fill(0.25);
    const result = new Float32Array(8);
    r.run('dep_order', xData, w1Data, w2Data, result);

    const gate = 2.0;
    const silu_gate = gate / (1 + Math.exp(-gate));
    const up = 1.0;
    const expected = silu_gate * up;
    for (let i = 0; i < 8; i++) {
      expect(result[i]).toBeCloseTo(expected, 3);
    }
  });

  it('parallel matmuls feeding fused elementwise produce correct results', () => {
    const x = new TensorType([2, 3], ScalarType.F32);
    const w1 = new TensorType([3, 2], ScalarType.F32);
    const w2 = new TensorType([3, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('par_mm', [x, w1, w2], [out], (b, args) => {
      const a = b.matmul(args[0], args[1]).getResult(0);
      const bv = b.matmul(args[0], args[2]).getResult(0);
      const sum = b.add(a, bv).getResult(0);
      b.returnOp([sum]);
    });

    const r = compile(func);
    const xData = new Float32Array([1, 0, 0, 0, 1, 0]);
    const w1Data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const w2Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    const result = new Float32Array(4);
    r.run('par_mm', xData, w1Data, w2Data, result);
    expect(Array.from(result)).toEqual([11, 22, 33, 44]);
  });
});

describe('complex chain: reshape → matmul → neg', () => {
  it('flattens 2x2 to 1x4, matmul with 4x2, negate', () => {
    const inp = new TensorType([2, 2], ScalarType.F32);
    const w = new TensorType([4, 2], ScalarType.F32);
    const out = new TensorType([1, 2], ScalarType.F32);

    const func = buildFunction('chain', [inp, w], [out], (b, args) => {
      const flat = b.reshape(args[0], [1, 4]);
      const mm = b.matmul(flat.getResult(0), args[1]);
      b.returnOp([b.neg(mm.getResult(0)).getResult(0)]);
    });

    const r = compile(func);
    const x = new Float32Array([1, 2, 3, 4]);
    const ww = new Float32Array([1, 0, 1, 0, 1, 0, 1, 0]);
    const o = new Float32Array(2);
    r.run('chain', x, ww, o);
    expect(o[0]).toBe(-10);
    expect(o[1]).toBeCloseTo(0);
  });
});
