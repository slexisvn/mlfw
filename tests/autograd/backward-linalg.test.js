import { describe, it, expect } from 'vitest';
import { tensor, sum, matmul } from '../../src/index.js';

function gradArr(t) {
  return t.grad.toArray();
}

describe('matmul backward 2D x 2D', () => {
  it('dC/dA = grad @ B^T, dC/dB = A^T @ grad', () => {
    const A = tensor([[1, 2], [3, 4]], { requiresGrad: true });
    const B = tensor([[5, 6], [7, 8]], { requiresGrad: true });
    sum(matmul(A, B)).backward();

    const gA = gradArr(A);
    expect(gA[0][0]).toBeCloseTo(5 + 6);
    expect(gA[0][1]).toBeCloseTo(7 + 8);
    expect(gA[1][0]).toBeCloseTo(5 + 6);
    expect(gA[1][1]).toBeCloseTo(7 + 8);

    const gB = gradArr(B);
    expect(gB[0][0]).toBeCloseTo(1 + 3);
    expect(gB[0][1]).toBeCloseTo(1 + 3);
    expect(gB[1][0]).toBeCloseTo(2 + 4);
    expect(gB[1][1]).toBeCloseTo(2 + 4);
  });
});

describe('matmul 1D x 2D (vector @ matrix)', () => {
  it('produces a 1D [n] result with correct values', () => {
    const v = tensor([1, 2, 3]);
    const M = tensor([[1, 2], [3, 4], [5, 6]]);
    const out = matmul(v, M);
    expect(out.shape).toEqual([2]);
    expect([...out.data]).toEqual([1 * 1 + 2 * 3 + 3 * 5, 1 * 2 + 2 * 4 + 3 * 6]);
  });

  it('computes gradients for vector-matrix multiply', () => {
    const v = tensor([1, 2, 3], { requiresGrad: true });
    const M = tensor([[1, 2], [3, 4], [5, 6]], { requiresGrad: true });
    sum(matmul(v, M)).backward();

    const gv = [...v.grad.data];
    expect(gv[0]).toBeCloseTo(1 + 2);
    expect(gv[1]).toBeCloseTo(3 + 4);
    expect(gv[2]).toBeCloseTo(5 + 6);

    const gM = gradArr(M);
    expect(gM[0][0]).toBeCloseTo(1);
    expect(gM[1][0]).toBeCloseTo(2);
    expect(gM[2][0]).toBeCloseTo(3);
  });
});

describe('matmul backward 2D x 1D', () => {
  it('computes gradients for matrix-vector multiply', () => {
    const A = tensor([[1, 2], [3, 4]], { requiresGrad: true });
    const v = tensor([1, 1], { requiresGrad: true });
    sum(matmul(A, v)).backward();

    const gA = gradArr(A);
    expect(gA[0][0]).toBeCloseTo(1);
    expect(gA[0][1]).toBeCloseTo(1);
    expect(gA[1][0]).toBeCloseTo(1);
    expect(gA[1][1]).toBeCloseTo(1);

    const gv = [...v.grad.data];
    expect(gv[0]).toBeCloseTo(1 + 3);
    expect(gv[1]).toBeCloseTo(2 + 4);
  });
});
