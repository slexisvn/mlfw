import { Metric } from './metric.js';
import type { MetricValue } from '../types.js';

export class MeanMetric extends Metric {
  private _sum: number;
  private _count: number;

  constructor() {
    super();
    this._sum = 0;
    this._count = 0;
  }

  update(value: MetricValue, weight = 1): void {
    const v = typeof value === 'number' ? value : value.item();
    this._sum += (v as number) * weight;
    this._count += weight;
  }

  compute(): number {
    return this._count === 0 ? 0 : this._sum / this._count;
  }

  reset(): void {
    super.reset();
    this._sum = 0;
    this._count = 0;
  }
}

export class SumMetric extends Metric {
  private _sum: number;

  constructor() {
    super();
    this._sum = 0;
  }

  update(value: MetricValue): void {
    this._sum += (typeof value === 'number' ? value : value.item()) as number;
  }

  compute(): number {
    return this._sum;
  }

  reset(): void {
    super.reset();
    this._sum = 0;
  }
}
