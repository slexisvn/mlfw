import { hostMatrix, hostColumns, toHostTensor } from '../tensor/utils/host_matrix.js';
import { _dispatch } from '../tensor/ops/ops.js';
import { tensor } from '../tensor/factory/from_ops.js';
import type { DType, NumericTypedArray } from '../tensor/types/dtype.js';
import type { HostMatrix, HostVector, MLTensor } from './types.js';

export function takeRows(X: MLTensor, idxArr: readonly number[]): MLTensor {
  const idx = tensor(Int32Array.from(idxArr), { shape: [idxArr.length], dtype: 'i32', device: X.device });
  return _dispatch('index_select', X, idx, 0) as MLTensor;
}

export function matrixOf(X: MLTensor): HostMatrix {
  return hostMatrix(X);
}

export function vectorOf(y: MLTensor): HostVector {
  const c = hostColumns(y);
  if (!c.wasVector && c.cols !== 1) throw new Error('ml: expected a 1-D target or single-column matrix');
  return { data: c.data, n: c.rows };
}

export function matrix(data: NumericTypedArray | ArrayLike<number> | number[], rows: number, cols: number, dtype?: DType): MLTensor {
  return toHostTensor(data, [rows, cols], dtype ?? 'f32') as MLTensor;
}

export function vector(data: NumericTypedArray | ArrayLike<number> | number[], n: number, dtype?: DType): MLTensor {
  return toHostTensor(data, [n], dtype ?? 'f32') as MLTensor;
}

export function column(m: HostMatrix, j: number, out?: Float64Array): Float64Array {
  const dst = out ?? new Float64Array(m.rows);
  for (let i = 0; i < m.rows; i++) dst[i] = m.data[i * m.cols + j];
  return dst;
}

export function columnMeans(m: HostMatrix): Float64Array {
  const mean = new Float64Array(m.cols);
  for (let j = 0; j < m.cols; j++) {
    let s = 0;
    for (let i = 0; i < m.rows; i++) s += m.data[i * m.cols + j];
    mean[j] = s / m.rows;
  }
  return mean;
}

export function encodeLabels(data: ArrayLike<number>, n: number): { y: Int32Array; classes: number[] } {
  const classes: number[] = [];
  const lookup = new Map<number, number>();
  const y = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const v = data[i];
    let idx = lookup.get(v);
    if (idx === undefined) {
      idx = classes.length;
      lookup.set(v, idx);
      classes.push(v);
    }
    y[i] = idx;
  }
  return { y, classes };
}
