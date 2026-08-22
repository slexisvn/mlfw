import { Tensor } from '../tensor/core/tensor.js';
import { float64From } from '../util/numeric_array.js';
import { tensorToContiguous } from '../dispatcher/jit_dispatch.js';
import { CPU_DEVICE } from '../tensor/types/device.js';
import type { HostGrid, HostVector, NumericMatrixInput, NumericVectorInput } from './types.js';

export function hostVector(x: NumericVectorInput): HostVector {
  if (x instanceof Tensor) {
    return { data: float64From(tensorToContiguous(x)), dtype: x.dtype, device: x.device };
  }
  if (typeof x === 'number') {
    return { data: Float64Array.of(x), dtype: 'f64', device: CPU_DEVICE };
  }
  return { data: Float64Array.from(x), dtype: 'f64', device: CPU_DEVICE };
}

export function hostGrid(x: NumericMatrixInput): HostGrid {
  if (x instanceof Tensor) {
    if (x.ndim !== 2) throw new Error(`numeric: expected a 2-D table, got ${x.ndim}-D`);
    const [rows, cols] = x.shape;
    return { data: float64From(tensorToContiguous(x)), rows, cols, dtype: x.dtype, device: x.device };
  }
  const rows = x.length;
  const cols = x[0].length;
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) data[i * cols + j] = x[i][j];
  }
  return { data, rows, cols, dtype: 'f64', device: CPU_DEVICE };
}
