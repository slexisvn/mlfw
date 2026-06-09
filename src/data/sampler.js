export class Sampler {
  *[Symbol.iterator]() {
    throw new Error('Subclass must implement [Symbol.iterator]()');
  }
}

export class SequentialSampler extends Sampler {
  constructor(dataSource) {
    super();
    this._dataSource = dataSource;
  }

  *[Symbol.iterator]() {
    const n = this._dataSource.length;
    for (let i = 0; i < n; i++) {
      yield i;
    }
  }
}

export class RandomSampler extends Sampler {
  constructor(dataSource) {
    super();
    this._dataSource = dataSource;
  }

  *[Symbol.iterator]() {
    const n = this._dataSource.length;
    const indices = new Int32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
    for (let i = 0; i < n; i++) {
      yield indices[i];
    }
  }
}

export class BatchSampler extends Sampler {
  constructor(sampler, batchSize, dropLast = false) {
    super();
    this._sampler = sampler;
    this._batchSize = batchSize;
    this._dropLast = dropLast;
  }

  *[Symbol.iterator]() {
    let batch = [];
    for (const idx of this._sampler) {
      batch.push(idx);
      if (batch.length === this._batchSize) {
        yield batch;
        batch = [];
      }
    }
    if (batch.length > 0 && !this._dropLast) {
      yield batch;
    }
  }
}
