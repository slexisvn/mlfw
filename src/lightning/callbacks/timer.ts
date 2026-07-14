import { Callback } from './callback.js';

export class Timer extends Callback {
  private _fitStartTime: number;
  private _epochStartTime: number;
  private _epochDurations: number[];
  private _validationDurations: number[];
  private _totalTrainingTime: number;
  private _valStartTime: number;

  constructor() {
    super();
    this._fitStartTime = 0;
    this._epochStartTime = 0;
    this._epochDurations = [];
    this._validationDurations = [];
    this._totalTrainingTime = 0;
    this._valStartTime = 0;
  }

  get epochDurations(): number[] { return this._epochDurations; }
  get validationDurations(): number[] { return this._validationDurations; }
  get totalTrainingTime(): number { return this._totalTrainingTime; }

  onFitStart(_trainer: unknown, _model: unknown): void {
    this._fitStartTime = performance.now();
  }

  onFitEnd(_trainer: unknown, _model: unknown): void {
    this._totalTrainingTime = (performance.now() - this._fitStartTime) / 1e3;
  }

  onTrainEpochStart(_trainer: unknown, _model: unknown): void {
    this._epochStartTime = performance.now();
  }

  onTrainEpochEnd(_trainer: unknown, _model: unknown): void {
    this._epochDurations.push((performance.now() - this._epochStartTime) / 1e3);
  }

  onValidationStart(_trainer: unknown, _model: unknown): void {
    this._valStartTime = performance.now();
  }

  onValidationEnd(_trainer: unknown, _model: unknown): void {
    this._validationDurations.push((performance.now() - this._valStartTime) / 1e3);
  }
}
