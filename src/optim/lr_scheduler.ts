import type { Optimizer } from './optimizer.js';

export class LRScheduler {
  protected readonly _optimizer: Optimizer;
  protected readonly _baseLRs: number[];
  protected _lastEpoch: number;
  protected _lastLR: number[] | null;

  constructor(optimizer: Optimizer, lastEpoch = -1) {
    this._optimizer = optimizer;
    this._baseLRs = optimizer.paramGroups.map(g => g.lr as number);
    this._lastEpoch = lastEpoch;
    this._lastLR = null;
  }

  _init(): void {
    this.step();
  }

  getLR(): number[] {
    throw new Error(`${this.constructor.name}.getLR() not implemented`);
  }

  getLastLR(): number[] | null {
    return this._lastLR;
  }

  step(): void {
    this._lastEpoch++;
    const lrs = this.getLR();
    this._lastLR = lrs;
    const groups = this._optimizer.paramGroups;
    for (let i = 0; i < groups.length; i++) {
      groups[i].lr = lrs[i];
    }
  }
}

export class StepLR extends LRScheduler {
  private readonly _stepSize: number;
  private readonly _gamma: number;

  constructor(optimizer: Optimizer, stepSize: number, gamma = 0.1, lastEpoch = -1) {
    super(optimizer, lastEpoch);
    this._stepSize = stepSize;
    this._gamma = gamma;
    this._init();
  }

  override getLR(): number[] {
    const factor = Math.pow(this._gamma, Math.floor(this._lastEpoch / this._stepSize));
    return this._baseLRs.map(base => base * factor);
  }
}

export class CosineAnnealingLR extends LRScheduler {
  private readonly _tMax: number;
  private readonly _etaMin: number;

  constructor(optimizer: Optimizer, tMax: number, etaMin = 0, lastEpoch = -1) {
    super(optimizer, lastEpoch);
    this._tMax = tMax;
    this._etaMin = etaMin;
    this._init();
  }

  override getLR(): number[] {
    const ratio = (1 + Math.cos(Math.PI * this._lastEpoch / this._tMax)) / 2;
    return this._baseLRs.map(base => this._etaMin + (base - this._etaMin) * ratio);
  }
}

export class ReduceLROnPlateau {
  private readonly _optimizer: Optimizer;
  private readonly _mode: string;
  private readonly _factor: number;
  private readonly _patience: number;
  private readonly _threshold: number;
  private readonly _thresholdMode: string;
  private readonly _cooldown: number;
  private readonly _minLR: number;
  private readonly _eps: number;
  private _best: number;
  private _numBadEpochs: number;
  private _cooldownCounter: number;

  constructor(optimizer: Optimizer, {
    mode = 'min',
    factor = 0.1,
    patience = 10,
    threshold = 1e-4,
    thresholdMode = 'rel',
    cooldown = 0,
    minLR = 0,
    eps = 1e-8,
  }: {
    mode?: string;
    factor?: number;
    patience?: number;
    threshold?: number;
    thresholdMode?: string;
    cooldown?: number;
    minLR?: number;
    eps?: number;
  } = {}) {
    this._optimizer = optimizer;
    this._mode = mode;
    this._factor = factor;
    this._patience = patience;
    this._threshold = threshold;
    this._thresholdMode = thresholdMode;
    this._cooldown = cooldown;
    this._minLR = minLR;
    this._eps = eps;
    this._best = mode === 'min' ? Infinity : -Infinity;
    this._numBadEpochs = 0;
    this._cooldownCounter = 0;
  }

  step(metric?: number): void {
    if (metric === undefined) throw new Error('ReduceLROnPlateau.step() requires a metric value');
    if (this._cooldownCounter > 0) {
      this._cooldownCounter--;
      this._numBadEpochs = 0;
    }
    if (this._isBetter(metric)) {
      this._best = metric;
      this._numBadEpochs = 0;
    } else {
      this._numBadEpochs++;
    }
    if (this._numBadEpochs > this._patience) {
      this._reduceAllLRs();
      this._cooldownCounter = this._cooldown;
      this._numBadEpochs = 0;
    }
  }

  _isBetter(current: number): boolean {
    if (this._mode === 'min') {
      return this._thresholdMode === 'rel'
        ? current < this._best * (1 - this._threshold)
        : current < this._best - this._threshold;
    }
    return this._thresholdMode === 'rel'
      ? current > this._best * (1 + this._threshold)
      : current > this._best + this._threshold;
  }

  _reduceAllLRs(): void {
    for (const group of this._optimizer.paramGroups) {
      const lr = group.lr as number;
      const newLR = Math.max(lr * this._factor, this._minLR);
      if (lr - newLR > this._eps) {
        group.lr = newLR;
      }
    }
  }
}
