import type { Metric } from './metric.js';
import type { MetricResult, TensorLike } from '../types.js';

type MetricMap = Record<string, Metric>;

export class MetricCollection {
  private _metrics: Map<string, Metric>;

  constructor(metrics: MetricMap = {}) {
    this._metrics = new Map();
    const keys = Object.keys(metrics);
    for (let i = 0; i < keys.length; i++) {
      this._metrics.set(keys[i], metrics[keys[i]]);
    }
  }

  add(name: string, metric: Metric): this {
    this._metrics.set(name, metric);
    return this;
  }

  update(preds: TensorLike, target: TensorLike): void {
    for (const [, metric] of this._metrics) {
      metric.update(preds, target);
    }
  }

  compute(): Record<string, MetricResult> {
    const result: Record<string, MetricResult> = {};
    for (const [name, metric] of this._metrics) {
      result[name] = metric.compute();
    }
    return result;
  }

  reset(): void {
    for (const [, metric] of this._metrics) {
      metric.reset();
    }
  }

  forward(preds: TensorLike, target: TensorLike): Record<string, MetricResult> {
    this.update(preds, target);
    return this.compute();
  }

  get(name: string): Metric | undefined {
    return this._metrics.get(name);
  }

  has(name: string): boolean {
    return this._metrics.has(name);
  }

  get size(): number {
    return this._metrics.size;
  }

  [Symbol.iterator](): MapIterator<[string, Metric]> {
    return this._metrics.entries();
  }
}
