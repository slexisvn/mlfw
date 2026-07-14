import type { MetricResult } from '../types.js';

export class Metric {
  protected _computed: MetricResult | null;

  constructor() {
    this._computed = null;
  }

  update(_preds: unknown, _target: unknown): void {
    throw new Error(`${this.constructor.name}.update() not implemented`);
  }

  compute(): MetricResult {
    throw new Error(`${this.constructor.name}.compute() not implemented`);
  }

  reset(): void {
    this._computed = null;
  }

  forward(preds: unknown, target: unknown): MetricResult {
    this.update(preds, target);
    this._computed = this.compute();
    return this._computed;
  }

  get value(): MetricResult | null {
    return this._computed;
  }
}
