import type { HyperparameterRecord, LoggerOptions, NumericMetricRecord } from '../types.js';

export class Logger {
  protected _name: string;
  protected _version: number | null;

  constructor({ name = 'default', version = 0 }: LoggerOptions = {}) {
    this._name = name;
    this._version = version;
  }

  get name(): string {
    return this._name;
  }

  get version(): number | null {
    return this._version;
  }

  logMetrics(_metrics: NumericMetricRecord, _step: number): void {
    throw new Error(`${this.constructor.name}.logMetrics() not implemented`);
  }

  logHyperparams(_params: HyperparameterRecord): void {
    throw new Error(`${this.constructor.name}.logHyperparams() not implemented`);
  }

  finalize(): void {}
}
