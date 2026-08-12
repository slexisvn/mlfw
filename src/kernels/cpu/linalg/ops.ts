import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { scalar } from '../../../tensor/factory/from_ops.js';
import { solveHost, detHost } from './lu.js';
import { choleskyHost } from './cholesky.js';
import { eighHost } from './jacobi_eigh.js';
import { svdHost } from './svd.js';
import { lstsqHost, pinvHost } from './pinv.js';
import { covHost } from './stats.js';
import type { Tensor } from '../../../tensor/core/tensor.js';
import type { HostMatrix } from '../../../tensor/utils/host_matrix.js';

function squareMatrix(a: Tensor, op: string): HostMatrix {
  const m = hostMatrix(a);
  if (m.rows !== m.cols) throw new Error(`linalg.${op}: matrix must be square`);
  return m;
}

export function cpuCholesky(_keySet: unknown, a: Tensor): Tensor {
  const { data, rows } = squareMatrix(a, 'cholesky');
  return toHostTensor(choleskyHost(data, rows), [rows, rows], a.dtype, a.device);
}

export function cpuSolve(_keySet: unknown, a: Tensor, b: Tensor): Tensor {
  const { data, rows } = squareMatrix(a, 'solve');
  const rhs = hostColumns(b);
  if (rhs.rows !== rows) throw new Error('linalg.solve: right-hand side rows must match matrix');
  const x = solveHost(data, rows, rhs.data, rhs.cols);
  return toHostTensor(x, rhs.wasVector ? [rows] : [rows, rhs.cols], a.dtype, a.device);
}

export function cpuLstsq(_keySet: unknown, a: Tensor, b: Tensor): Tensor {
  const { data, rows, cols } = hostMatrix(a);
  const rhs = hostColumns(b);
  if (rhs.rows !== rows) throw new Error('linalg.lstsq: right-hand side rows must match matrix');
  const x = lstsqHost(data, rows, cols, rhs.data, rhs.cols);
  return toHostTensor(x, rhs.wasVector ? [cols] : [cols, rhs.cols], a.dtype, a.device);
}

export function cpuInv(_keySet: unknown, a: Tensor): Tensor {
  const { data, rows } = squareMatrix(a, 'inv');
  const eye = new Float64Array(rows * rows);
  for (let i = 0; i < rows; i++) eye[i * rows + i] = 1;
  const x = solveHost(data, rows, eye, rows);
  return toHostTensor(x, [rows, rows], a.dtype, a.device);
}

export function cpuPinv(_keySet: unknown, a: Tensor): Tensor {
  const { data, rows, cols } = hostMatrix(a);
  return toHostTensor(pinvHost(data, rows, cols), [cols, rows], a.dtype, a.device);
}

export function cpuDet(_keySet: unknown, a: Tensor): Tensor {
  const { data, rows } = squareMatrix(a, 'det');
  return scalar(detHost(data, rows), { dtype: a.dtype });
}

export function cpuCov(_keySet: unknown, a: Tensor): Tensor {
  const { data, rows, cols } = hostMatrix(a);
  return toHostTensor(covHost(data, rows, cols), [cols, cols], a.dtype, a.device);
}

export function cpuEigh(_keySet: unknown, a: Tensor): Tensor[] {
  const { data, rows } = squareMatrix(a, 'eigh');
  const { values, vectors } = eighHost(data, rows);
  return [toHostTensor(values, [rows], a.dtype, a.device), toHostTensor(vectors, [rows, rows], a.dtype, a.device)];
}

export function cpuSvd(_keySet: unknown, a: Tensor): Tensor[] {
  const { data, rows, cols } = hostMatrix(a);
  const { U, S, V, k } = svdHost(data, rows, cols);
  return [toHostTensor(U, [rows, k], a.dtype, a.device), toHostTensor(S, [k], a.dtype, a.device), toHostTensor(V, [cols, k], a.dtype, a.device)];
}
