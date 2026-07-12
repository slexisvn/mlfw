import { select } from '../tensor/ops/ops.js';

export class Dataset {
  get length() {
    throw new Error('Subclass must implement get length()');
  }

  get(index) {
    throw new Error('Subclass must implement get(index)');
  }

  *[Symbol.iterator]() {
    const n = this.length;
    for (let i = 0; i < n; i++) {
      yield this.get(i);
    }
  }
}

export class TensorDataset extends Dataset {
  constructor(...tensors) {
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

  get length() {
    return this._length;
  }

  get(index) {
    const result = new Array(this._tensors.length);
    for (let i = 0; i < this._tensors.length; i++) {
      result[i] = select(this._tensors[i], 0, index);
    }
    return result;
  }
}

export class MapDataset extends Dataset {
  constructor(dataset, transform) {
    super();
    this._dataset = dataset;
    this._transform = transform;
  }

  get length() {
    return this._dataset.length;
  }

  get(index) {
    return this._transform(this._dataset.get(index));
  }
}
