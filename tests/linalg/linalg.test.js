import { describe, it, expect } from 'vitest';
import { tensor, linalg, matmul } from '../../src/index.js';

function close(a, b, tol = 1e-5) {
  expect(Math.abs(a - b)).toBeLessThan(tol);
}

describe('linalg.cholesky', () => {
  it('reconstructs A = L Lᵀ for SPD', () => {
    const A = tensor([[4, 2], [2, 3]]);
    const L = linalg.cholesky(A).toArray();
    const rec = [
      [L[0][0] * L[0][0], L[0][0] * L[1][0]],
      [L[1][0] * L[0][0], L[1][0] * L[1][0] + L[1][1] * L[1][1]],
    ];
    close(rec[0][0], 4); close(rec[0][1], 2); close(rec[1][1], 3);
  });
  it('throws on non-SPD', () => {
    expect(() => linalg.cholesky(tensor([[1, 2], [2, 1]]))).toThrow();
  });
});

describe('linalg.solve / inv / det', () => {
  it('solves A x = b', () => {
    const A = tensor([[3, 2], [1, 2]]);
    const x = linalg.solve(A, tensor([5, 5])).toArray();
    close(3 * x[0] + 2 * x[1], 5); close(x[0] + 2 * x[1], 5);
  });
  it('multi-RHS', () => {
    const A = tensor([[2, 0], [0, 4]]);
    const X = linalg.solve(A, tensor([[2, 4], [8, 4]])).toArray();
    close(X[0][0], 1); close(X[1][0], 2); close(X[0][1], 2); close(X[1][1], 1);
  });
  it('inv @ A = I', () => {
    const A = tensor([[4, 2], [2, 3]]);
    const inv = linalg.inv(A);
    const I = matmul(inv, A).toArray();
    close(I[0][0], 1); close(I[1][1], 1); close(I[0][1], 0); close(I[1][0], 0);
  });
  it('det', () => {
    close(linalg.det(tensor([[1, 2], [3, 4]])), -2);
  });
});

describe('linalg.eigh', () => {
  it('A v = λ v and orthonormal vectors', () => {
    const A = tensor([[2, 1], [1, 2]]);
    const { values, vectors } = linalg.eigh(A);
    const val = values.toArray();
    const V = vectors.toArray();
    close(val[0], 1); close(val[1], 3);
    const dot = V[0][0] * V[0][1] + V[1][0] * V[1][1];
    close(dot, 0);
    const Av0 = [2 * V[0][0] + 1 * V[1][0], 1 * V[0][0] + 2 * V[1][0]];
    close(Av0[0], val[0] * V[0][0]); close(Av0[1], val[0] * V[1][0]);
  });
});

describe('linalg.svd', () => {
  it('reconstructs U diag(S) Vᵀ ≈ A (m>n)', () => {
    const A = [[1, 2], [3, 4], [5, 6]];
    const { U, S, V } = linalg.svd(tensor(A));
    const u = U.toArray(); const s = S.toArray(); const v = V.toArray();
    const k = s.length;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 2; j++) {
        let rec = 0;
        for (let c = 0; c < k; c++) rec += u[i][c] * s[c] * v[j][c];
        close(rec, A[i][j], 1e-4);
      }
    }
    expect(s[0]).toBeGreaterThanOrEqual(s[1]);
  });
  it('reconstructs (m<n)', () => {
    const A = [[1, 2, 3], [4, 5, 6]];
    const { U, S, V } = linalg.svd(tensor(A));
    const u = U.toArray(); const s = S.toArray(); const v = V.toArray();
    const k = s.length;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        let rec = 0;
        for (let c = 0; c < k; c++) rec += u[i][c] * s[c] * v[j][c];
        close(rec, A[i][j], 1e-4);
      }
    }
  });
});

describe('linalg.lstsq', () => {
  it('recovers exact solution (overdetermined consistent)', () => {
    const A = tensor([[1, 1], [1, 2], [1, 3]]);
    const y = tensor([6, 8, 10]);
    const w = linalg.lstsq(A, y).toArray();
    close(w[0], 4); close(w[1], 2);
  });
});
