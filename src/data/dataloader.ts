import { SequentialSampler, RandomSampler, BatchSampler } from './sampler.js';
import { defaultCollate } from './collate.js';
import type { Dataset } from './dataset.js';

type CollateFn<T, TBatch> = (samples: T[]) => TBatch;
type DataLoaderOptions<T, TBatch> = {
  batchSize?: number | null;
  shuffle?: boolean;
  sampler?: Iterable<number> | null;
  batchSampler?: Iterable<number[]> | null;
  dropLast?: boolean;
  collate?: CollateFn<T, TBatch>;
};

export class DataLoader<T = unknown, TBatch = unknown> implements Iterable<TBatch> {
  private readonly _dataset: Dataset<T>;
  private readonly _collate: CollateFn<T, TBatch>;
  private readonly _batchSampler: Iterable<number[]>;
  private readonly _batchSize: number | null;
  private readonly _dropLast: boolean | null;

  constructor(dataset: Dataset<T>, opts: DataLoaderOptions<T, TBatch> = {}) {
    this._dataset = dataset;
    this._collate = opts.collate ?? ((samples: T[]) => defaultCollate(samples) as TBatch);

    if (opts.batchSampler != null) {
      if (opts.batchSize != null || opts.shuffle || opts.sampler != null || opts.dropLast) {
        throw new Error(
          'batchSampler is mutually exclusive with batchSize, shuffle, sampler, and dropLast'
        );
      }
      this._batchSampler = opts.batchSampler;
      this._batchSize = null;
      this._dropLast = null;
    } else {
      const batchSize = opts.batchSize ?? 1;
      const dropLast = opts.dropLast ?? false;
      this._batchSize = batchSize;
      this._dropLast = dropLast;

      let sampler: Iterable<number>;
      if (opts.sampler != null) {
        if (opts.shuffle) {
          throw new Error('sampler and shuffle are mutually exclusive');
        }
        sampler = opts.sampler;
      } else {
        sampler = opts.shuffle
          ? new RandomSampler(dataset)
          : new SequentialSampler(dataset);
      }

      this._batchSampler = new BatchSampler(sampler, batchSize, dropLast);
    }
  }

  get dataset(): Dataset<T> {
    return this._dataset;
  }

  get length(): number {
    const n = this._dataset.length;
    if (this._batchSize == null) {
      let count = 0;
      for (const _ of this._batchSampler) count++;
      return count;
    }
    return this._dropLast
      ? Math.floor(n / this._batchSize)
      : Math.ceil(n / this._batchSize);
  }

  *[Symbol.iterator](): Generator<TBatch> {
    for (const indices of this._batchSampler) {
      const samples = new Array<T>(indices.length);
      for (let i = 0; i < indices.length; i++) {
        samples[i] = this._dataset.get(indices[i]);
      }
      yield this._collate(samples);
    }
  }
}
