export class LRScheduler {
  constructor(optimizer, lastEpoch = -1) {
    this._optimizer = optimizer;
    this._baseLRs = optimizer.paramGroups.map(g => g.lr);
    this._lastEpoch = lastEpoch;
    this._lastLR = null;
  }

  _init() {
    this.step();
  }

  getLR() {
    throw new Error(`${this.constructor.name}.getLR() not implemented`);
  }

  getLastLR() {
    return this._lastLR;
  }

  step() {
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
  constructor(optimizer, stepSize, gamma = 0.1, lastEpoch = -1) {
    super(optimizer, lastEpoch);
    this._stepSize = stepSize;
    this._gamma = gamma;
    this._init();
  }

  getLR() {
    const factor = Math.pow(this._gamma, Math.floor(this._lastEpoch / this._stepSize));
    return this._baseLRs.map(base => base * factor);
  }
}

export class CosineAnnealingLR extends LRScheduler {
  constructor(optimizer, tMax, etaMin = 0, lastEpoch = -1) {
    super(optimizer, lastEpoch);
    this._tMax = tMax;
    this._etaMin = etaMin;
    this._init();
  }

  getLR() {
    const ratio = (1 + Math.cos(Math.PI * this._lastEpoch / this._tMax)) / 2;
    return this._baseLRs.map(base => this._etaMin + (base - this._etaMin) * ratio);
  }
}

export class ReduceLROnPlateau {
  constructor(optimizer, {
    mode = 'min',
    factor = 0.1,
    patience = 10,
    threshold = 1e-4,
    thresholdMode = 'rel',
    cooldown = 0,
    minLR = 0,
    eps = 1e-8,
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

  step(metric) {
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

  _isBetter(current) {
    if (this._mode === 'min') {
      return this._thresholdMode === 'rel'
        ? current < this._best * (1 - this._threshold)
        : current < this._best - this._threshold;
    }
    return this._thresholdMode === 'rel'
      ? current > this._best * (1 + this._threshold)
      : current > this._best + this._threshold;
  }

  _reduceAllLRs() {
    for (const group of this._optimizer.paramGroups) {
      const newLR = Math.max(group.lr * this._factor, this._minLR);
      if (group.lr - newLR > this._eps) {
        group.lr = newLR;
      }
    }
  }
}
