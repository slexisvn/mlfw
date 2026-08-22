import { Callback } from './callback.js';
import type { TrainerLike } from '../types.js';

type EarlyStoppingMode = 'min' | 'max';
type CompareFn = (current: number, best: number) => boolean;
type EarlyStoppingOptions = {
  monitor?: string;
  patience?: number;
  mode?: EarlyStoppingMode;
  minDelta?: number;
  checkOnTrainEpochEnd?: boolean;
};

export class EarlyStopping extends Callback {
  private _monitor: string;
  private _patience: number;
  private _checkOnTrainEpochEnd: boolean;
  private _waitCount: number;
  private _bestScore: number | null;
  private _compareFn: CompareFn;

  constructor({
    monitor = 'val_loss',
    patience = 3,
    mode = 'min',
    minDelta = 0,
    checkOnTrainEpochEnd = false,
  }: EarlyStoppingOptions = {}) {
    super();
    this._monitor = monitor;
    this._patience = patience;
    this._checkOnTrainEpochEnd = checkOnTrainEpochEnd;
    this._waitCount = 0;
    this._bestScore = null;
    this._compareFn = mode === 'min'
      ? (current, best) => current < best - minDelta
      : (current, best) => current > best + minDelta;
  }

  get monitor(): string { return this._monitor; }
  get patience(): number { return this._patience; }
  get bestScore(): number | null { return this._bestScore; }
  get waitCount(): number { return this._waitCount; }

  onValidationEnd(trainer: TrainerLike, _model?: unknown): void {
    if (this._checkOnTrainEpochEnd) return;
    this._check(trainer);
  }

  onTrainEpochEnd(trainer: TrainerLike, _model?: unknown): void {
    if (!this._checkOnTrainEpochEnd) return;
    this._check(trainer);
  }

  private _check(trainer: TrainerLike): void {
    const metrics = trainer.state.epochMetrics!.computeAll();
    const current = metrics[this._monitor];
    if (current === undefined) return;

    if (this._bestScore === null || this._compareFn(current, this._bestScore)) {
      this._bestScore = current;
      this._waitCount = 0;
      return;
    }

    this._waitCount++;
    if (this._waitCount >= this._patience) {
      trainer.shouldStop = true;
    }
  }

  reset(): void {
    this._waitCount = 0;
    this._bestScore = null;
  }
}
