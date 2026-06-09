import { Logger } from './logger.js';

export class ConsoleLogger extends Logger {
  constructor(opts = {}) {
    super(opts);
    this._logFrequency = opts.logFrequency || 1;
    this._callCount = 0;
  }

  logMetrics(metrics, step) {
    this._callCount++;
    if (this._callCount % this._logFrequency !== 0) return;
    const parts = [`[step ${step}]`];
    const keys = Object.keys(metrics).sort();
    for (let i = 0; i < keys.length; i++) {
      const v = metrics[keys[i]];
      parts.push(`${keys[i]}: ${formatNumber(v)}`);
    }
    console.log(parts.join(' | '));
  }

  logHyperparams(params) {
    const parts = ['[hyperparams]'];
    const keys = Object.keys(params).sort();
    for (let i = 0; i < keys.length; i++) {
      parts.push(`${keys[i]}: ${params[keys[i]]}`);
    }
    console.log(parts.join(' | '));
  }
}

function formatNumber(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
  return v.toFixed(4);
}
