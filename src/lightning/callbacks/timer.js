import { Callback } from './callback.js';

export class Timer extends Callback {
  constructor() {
    super();
    this._fitStartTime = 0;
    this._epochStartTime = 0;
    this._epochDurations = [];
    this._validationDurations = [];
    this._totalTrainingTime = 0;
    this._valStartTime = 0;
  }

  get epochDurations() { return this._epochDurations; }
  get validationDurations() { return this._validationDurations; }
  get totalTrainingTime() { return this._totalTrainingTime; }

  onFitStart(_trainer, _model) {
    this._fitStartTime = performance.now();
  }

  onFitEnd(_trainer, _model) {
    this._totalTrainingTime = (performance.now() - this._fitStartTime) / 1e3;
  }

  onTrainEpochStart(_trainer, _model) {
    this._epochStartTime = performance.now();
  }

  onTrainEpochEnd(_trainer, _model) {
    this._epochDurations.push((performance.now() - this._epochStartTime) / 1e3);
  }

  onValidationStart(_trainer, _model) {
    this._valStartTime = performance.now();
  }

  onValidationEnd(_trainer, _model) {
    this._validationDurations.push((performance.now() - this._valStartTime) / 1e3);
  }
}
