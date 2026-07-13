import { select } from '../tensor/ops/ops.js';
import type { Tensor } from '../tensor/core/tensor.js';

export class Dataset<T = unknown> implements Iterable<T> {
  get length(): number {
    throw new Error('Subclass must implement get length()');
  }

  get(index: number): T {
    throw new Error('Subclass must implement get(index)');
  }

  *[Symbol.iterator](): Generator<T> {
    const n = this.length;
    for (let i = 0; i < n; i++) {
      yield this.get(i);
    }
  }
}

export class TensorDataset extends Dataset<Tensor[]> {
  private readonly _tensors: Tensor[];
  private readonly _length: number;

  constructor(...tensors: Tensor[]) {
    super();
    if (tensors.length === 0) {
      throw new Error('TensorDataset requires at least one tensor');
    }
    const size = tensors[0].shape[0];
    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape[0] !== size) {
        throw new Error(
          `Size mismatch at dim 0: tensor 0 has ${size}, tensor ${i} has ${tensors[i].shape[0]}`
        );
      }
    }
    this._tensors = tensors;
    this._length = size;
  }

  get length(): number {
    return this._length;
  }

  get(index: number): Tensor[] {
    const result = new Array(this._tensors.length);
    for (let i = 0; i < this._tensors.length; i++) {
      result[i] = select(this._tensors[i], 0, index);
    }
    return result;
  }
}

export class MapDataset<TIn, TOut> extends Dataset<TOut> {
  private readonly _dataset: Dataset<TIn>;
  private readonly _transform: (sample: TIn) => TOut;

  constructor(dataset: Dataset<TIn>, transform: (sample: TIn) => TOut) {
    super();
    this._dataset = dataset;
    this._transform = transform;
  }

  get length(): number {
    return this._dataset.length;
  }

  get(index: number): TOut {
    return this._transform(this._dataset.get(index));
  }
}
