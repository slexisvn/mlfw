type Sized = { length: number };

export class Sampler implements Iterable<number | number[]> {
  *[Symbol.iterator](): Generator<number | number[]> {
    throw new Error('Subclass must implement [Symbol.iterator]()');
  }
}

export class SequentialSampler extends Sampler implements Iterable<number> {
  private readonly _dataSource: Sized;

  constructor(dataSource: Sized) {
    super();
    this._dataSource = dataSource;
  }

  *[Symbol.iterator](): Generator<number> {
    const n = this._dataSource.length;
    for (let i = 0; i < n; i++) {
      yield i;
    }
  }
}

export class RandomSampler extends Sampler implements Iterable<number> {
  private readonly _dataSource: Sized;

  constructor(dataSource: Sized) {
    super();
    this._dataSource = dataSource;
  }

  *[Symbol.iterator](): Generator<number> {
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

export class BatchSampler extends Sampler implements Iterable<number[]> {
  readonly _dropLast: boolean;
  private readonly _sampler: Iterable<number>;
  private readonly _batchSize: number;

  constructor(sampler: Iterable<number>, batchSize: number, dropLast = false) {
    super();
    this._sampler = sampler;
    this._batchSize = batchSize;
    this._dropLast = dropLast;
  }

  *[Symbol.iterator](): Generator<number[]> {
    let batch: number[] = [];
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
