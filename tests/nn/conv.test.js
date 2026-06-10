import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { conv1d, conv2d } from '../../src/nn/functional/conv.js';

function flat(t) {
  return Array.from(t.contiguous().data);
}

describe('conv1d bias', () => {
  it('adds per-output-channel bias along the channel axis (not the length axis)', () => {
    const x = tensor([[[1, 2, 3, 4, 5]]]);
    const w = tensor([[[1, 1]], [[2, 2]]]);
    const b = tensor([10, 20]);
    const out = conv1d(x, w, b, 1, 0, 1, 1);
    expect(out.shape).toEqual([1, 2, 4]);
    expect(flat(out)).toEqual([
      13, 15, 17, 19,
      26, 30, 34, 38,
    ]);
  });

  it('matches conv1d without bias plus a manual channel offset', () => {
    const x = tensor([[[0, 1, 2, 3]]]);
    const w = tensor([[[1, 0]], [[0, 1]]]);
    const noBias = flat(conv1d(x, w, null, 1, 0, 1, 1));
    const withBias = flat(conv1d(x, w, tensor([5, 7]), 1, 0, 1, 1));
    const half = noBias.length / 2;
    for (let i = 0; i < half; i++) expect(withBias[i]).toBeCloseTo(noBias[i] + 5, 5);
    for (let i = half; i < noBias.length; i++) expect(withBias[i]).toBeCloseTo(noBias[i] + 7, 5);
  });
});

describe('conv2d bias', () => {
  it('adds per-output-channel bias along the channel axis', () => {
    const x = tensor([[[[1, 2], [3, 4]]]]);
    const w = tensor([[[[1]]], [[[2]]]]);
    const b = tensor([100, 200]);
    const out = conv2d(x, w, b, [1, 1], [[0, 0], [0, 0]]);
    expect(out.shape).toEqual([1, 2, 2, 2]);
    expect(flat(out)).toEqual([
      101, 102, 103, 104,
      202, 204, 206, 208,
    ]);
  });
});
