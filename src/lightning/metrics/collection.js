export class MetricCollection {
  constructor(metrics = {}) {
    this._metrics = new Map();
    const keys = Object.keys(metrics);
    for (let i = 0; i < keys.length; i++) {
      this._metrics.set(keys[i], metrics[keys[i]]);
    }
  }

  add(name, metric) {
    this._metrics.set(name, metric);
    return this;
  }

  update(preds, target) {
    for (const [, metric] of this._metrics) {
      metric.update(preds, target);
    }
  }

  compute() {
    const result = {};
    for (const [name, metric] of this._metrics) {
      result[name] = metric.compute();
    }
    return result;
  }

  reset() {
    for (const [, metric] of this._metrics) {
      metric.reset();
    }
  }

  forward(preds, target) {
    this.update(preds, target);
    return this.compute();
  }

  get(name) {
    return this._metrics.get(name);
  }

  has(name) {
    return this._metrics.has(name);
  }

  get size() {
    return this._metrics.size;
  }

  [Symbol.iterator]() {
    return this._metrics.entries();
  }
}
