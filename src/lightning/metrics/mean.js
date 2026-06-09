import { Metric } from './metric.js';

export class MeanMetric extends Metric {
  constructor() {
    super();
    this._sum = 0;
    this._count = 0;
  }

  update(value, weight = 1) {
    const v = typeof value === 'number' ? value : value.item();
    this._sum += v * weight;
    this._count += weight;
  }

  compute() {
    return this._count === 0 ? 0 : this._sum / this._count;
  }

  reset() {
    super.reset();
    this._sum = 0;
    this._count = 0;
  }
}

export class SumMetric extends Metric {
  constructor() {
    super();
    this._sum = 0;
  }

  update(value) {
    this._sum += typeof value === 'number' ? value : value.item();
  }

  compute() {
    return this._sum;
  }

  reset() {
    super.reset();
    this._sum = 0;
  }
}
