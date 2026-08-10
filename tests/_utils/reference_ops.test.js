import { describe, it, expect } from 'vitest';
import {
  matmul, transpose, slice, concat, broadcastTo, reduce,
  softmaxLastAxis, layerNormLastAxis, conv2d, sigmoid, tanh,
  gelu, GELU_SIGMOID_COEFF, resizeNearest, pool2d, batchNorm,
} from './reference_ops.js';

const arr = (a) => Array.from(a);

describe('reference matmul', () => {
  it('matches a hand-computed 2x3 @ 3x2 product', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    expect(arr(matmul(a, [2, 3], b, [3, 2]))).toEqual([58, 64, 139, 154]);
  });

  it('throws when the inner dimensions disagree', () => {
    expect(() => matmul(new Float32Array(6), [2, 3], new Float32Array(8), [4, 2])).toThrow(/shape mismatch/);
  });
});

describe('reference transpose', () => {
  it('swaps the axes of a 2x3 matrix', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect(arr(transpose(a, [2, 3], [1, 0]))).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it('permutes a 2x2x2 tensor to [1,0,2]', () => {
    const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(arr(transpose(a, [2, 2, 2], [1, 0, 2]))).toEqual([0, 1, 4, 5, 2, 3, 6, 7]);
  });
});

describe('reference slice', () => {
  it('extracts the middle column of a 2x3 matrix', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect(arr(slice(a, [2, 3], [0, 1], [2, 2]))).toEqual([2, 5]);
  });
});

describe('reference concat', () => {
  it('concatenates along the last axis', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([5, 6]);
    expect(arr(concat([a, b], [[2, 2], [2, 1]], 1))).toEqual([1, 2, 5, 3, 4, 6]);
  });

  it('concatenates along the first axis', () => {
    const a = new Float32Array([1, 2]);
    const b = new Float32Array([3, 4, 5, 6]);
    expect(arr(concat([a, b], [[1, 2], [2, 2]], 0))).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('reference broadcastTo', () => {
  it('stretches a row vector across rows', () => {
    expect(arr(broadcastTo(new Float32Array([1, 2, 3]), [1, 3], [2, 3]))).toEqual([1, 2, 3, 1, 2, 3]);
  });

  it('stretches a column vector across columns', () => {
    expect(arr(broadcastTo(new Float32Array([1, 2]), [2, 1], [2, 3]))).toEqual([1, 1, 1, 2, 2, 2]);
  });
});

describe('reference reduce', () => {
  it('sums over the last axis', () => {
    expect(arr(reduce(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], [1], 'sum'))).toEqual([6, 15]);
  });

  it('takes the max over the first axis', () => {
    expect(arr(reduce(new Float32Array([1, 9, 5, 2]), [2, 2], [0], 'max'))).toEqual([5, 9]);
  });
});

describe('reference softmax / layernorm', () => {
  it('softmax of equal logits is uniform and each row sums to 1', () => {
    const out = softmaxLastAxis(new Float32Array([2, 2, 2, 0, 0, 0]), [2, 3]);
    for (const v of out) expect(v).toBeCloseTo(1 / 3, 6);
  });

  it('softmax is shift-invariant', () => {
    const a = softmaxLastAxis(new Float32Array([1, 2, 3]), [1, 3]);
    const b = softmaxLastAxis(new Float32Array([1001, 1002, 1003]), [1, 3]);
    for (let i = 0; i < 3; i++) expect(a[i]).toBeCloseTo(b[i], 6);
  });

  it('layernorm gives zero mean and unit variance per row', () => {
    const out = layerNormLastAxis(new Float32Array([1, 2, 3, 4]), [1, 4], { eps: 0 });
    let mean = 0;
    for (const v of out) mean += v;
    expect(mean / 4).toBeCloseTo(0, 6);
    let varr = 0;
    for (const v of out) varr += v * v;
    expect(varr / 4).toBeCloseTo(1, 5);
  });

  it('layernorm applies gamma then beta per element', () => {
    const plain = layerNormLastAxis(new Float32Array([1, 2, 3, 4]), [1, 4], { eps: 0 });
    const scaled = layerNormLastAxis(new Float32Array([1, 2, 3, 4]), [1, 4], {
      gamma: new Float32Array([2, 2, 2, 2]),
      beta: new Float32Array([1, 1, 1, 1]),
      eps: 0,
    });
    for (let i = 0; i < 4; i++) expect(scaled[i]).toBeCloseTo(plain[i] * 2 + 1, 5);
  });
});

describe('reference gelu matches the framework sigmoid approximation', () => {
  it('gelu(x) equals x * sigmoid(1.702x), the form the decomposition pass emits', () => {
    const out = gelu(new Float32Array([-1, 0, 1, 2]));
    for (const [i, x] of [-1, 0, 1, 2].entries()) {
      expect(out[i]).toBeCloseTo(x / (1 + Math.exp(-GELU_SIGMOID_COEFF * x)), 6);
    }
  });
});

describe('reference resizeNearest', () => {
  it('doubles a 2x2 map by repeating each source pixel', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    expect(arr(resizeNearest(a, [1, 1, 2, 2], [4, 4]))).toEqual([
      1, 1, 2, 2,
      1, 1, 2, 2,
      3, 3, 4, 4,
      3, 3, 4, 4,
    ]);
  });
});

describe('reference pool2d', () => {
  it('2x2 max pool with stride 2 takes the largest of each quadrant', () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const { data, shape } = pool2d(x, [1, 1, 4, 4], 'max', [2, 2]);
    expect(shape).toEqual([1, 1, 2, 2]);
    expect(arr(data)).toEqual([6, 8, 14, 16]);
  });

  it('2x2 avg pool with stride 2 averages each quadrant', () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const { data } = pool2d(x, [1, 1, 4, 4], 'avg', [2, 2]);
    expect(arr(data)).toEqual([3.5, 5.5, 11.5, 13.5]);
  });

  it('global avg pool over the whole map equals the mean', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const { data, shape } = pool2d(x, [1, 1, 2, 2], 'avg', [2, 2]);
    expect(shape).toEqual([1, 1, 1, 1]);
    expect(arr(data)).toEqual([2.5]);
  });
});

describe('reference batchNorm', () => {
  it('normalizes per channel then applies gamma and beta', () => {
    const x = new Float32Array([1, 3, 10, 14]);
    const out = batchNorm(x, [1, 2, 1, 2], new Float32Array([2, 1]), new Float32Array([0, 5]),
      new Float32Array([2, 12]), new Float32Array([1, 4]), { eps: 0 });
    expect(out[0]).toBeCloseTo(-2, 6);
    expect(out[1]).toBeCloseTo(2, 6);
    expect(out[2]).toBeCloseTo(4, 6);
    expect(out[3]).toBeCloseTo(6, 6);
  });
});

describe('reference reduce mean', () => {
  it('averages over the reduced axis', () => {
    expect(arr(reduce(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], [1], 'mean'))).toEqual([2, 5]);
  });
});

describe('reference conv2d', () => {
  it('matches a hand-computed 1x1x3x3 input with a 2x2 kernel, stride 1, no padding', () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const w = new Float32Array([1, 0, 0, 1]);
    const { data, shape } = conv2d(x, [1, 1, 3, 3], w, [1, 1, 2, 2]);
    expect(shape).toEqual([1, 1, 2, 2]);
    expect(arr(data)).toEqual([6, 8, 12, 14]);
  });

  it('stride 2 halves the output extent', () => {
    const x = new Float32Array(16).fill(1);
    const w = new Float32Array([1, 1, 1, 1]);
    const { shape } = conv2d(x, [1, 1, 4, 4], w, [1, 1, 2, 2], { stride: 2 });
    expect(shape).toEqual([1, 1, 2, 2]);
  });

  it('padding 1 with a 3x3 kernel preserves the spatial extent', () => {
    const x = new Float32Array(9).fill(1);
    const w = new Float32Array(9).fill(1);
    const { data, shape } = conv2d(x, [1, 1, 3, 3], w, [1, 1, 3, 3], { pad: 1 });
    expect(shape).toEqual([1, 1, 3, 3]);
    expect(arr(data)).toEqual([4, 6, 4, 6, 9, 6, 4, 6, 4]);
  });

  it('groups=2 keeps each half of the channels independent', () => {
    const x = new Float32Array([1, 1, 2, 2]);
    const w = new Float32Array([3, 5]);
    const { data } = conv2d(x, [1, 2, 1, 2], w, [2, 1, 1, 1], { groups: 2 });
    expect(arr(data)).toEqual([3, 3, 10, 10]);
  });

  it('dilation 2 samples every other input position', () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const w = new Float32Array([1, 1, 1, 1]);
    const { data, shape } = conv2d(x, [1, 1, 3, 3], w, [1, 1, 2, 2], { dilation: 2 });
    expect(shape).toEqual([1, 1, 1, 1]);
    expect(arr(data)).toEqual([1 + 3 + 7 + 9]);
  });
});

describe('reference elementwise activations', () => {
  it('sigmoid(0)=0.5 and is symmetric about it', () => {
    const out = sigmoid(new Float32Array([0, 2, -2]));
    expect(out[0]).toBeCloseTo(0.5, 6);
    expect(out[1] + out[2]).toBeCloseTo(1, 6);
  });

  it('tanh(0)=0 and saturates at +/-1', () => {
    const out = tanh(new Float32Array([0, 20, -20]));
    expect(out[0]).toBeCloseTo(0, 6);
    expect(out[1]).toBeCloseTo(1, 6);
    expect(out[2]).toBeCloseTo(-1, 6);
  });
});
