import { SequentialSampler, RandomSampler, BatchSampler } from './sampler.js';
import { defaultCollate } from './collate.js';

export class DataLoader {
  constructor(dataset, opts = {}) {
    this._dataset = dataset;
    this._collate = opts.collate ?? defaultCollate;

    if (opts.batchSampler != null) {
      if (opts.batchSize != null || opts.shuffle || opts.sampler != null || opts.dropLast) {
        throw new Error(
          'batchSampler is mutually exclusive with batchSize, shuffle, sampler, and dropLast'
        );
      }
      this._batchSampler = opts.batchSampler;
      this._batchSize = null;
    } else {
      const batchSize = opts.batchSize ?? 1;
      const dropLast = opts.dropLast ?? false;
      this._batchSize = batchSize;

      let sampler;
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

  get dataset() {
    return this._dataset;
  }

  get length() {
    const n = this._dataset.length;
    if (this._batchSize == null) {
      let count = 0;
      for (const _ of this._batchSampler) count++;
      return count;
    }
    return this._batchSampler._dropLast
      ? Math.floor(n / this._batchSize)
      : Math.ceil(n / this._batchSize);
  }

  *[Symbol.iterator]() {
    for (const indices of this._batchSampler) {
      const samples = new Array(indices.length);
      for (let i = 0; i < indices.length; i++) {
        samples[i] = this._dataset.get(indices[i]);
      }
      yield this._collate(samples);
    }
  }
}
