export const Stage = Object.freeze({
  IDLE: 'idle',
  TRAINING: 'training',
  VALIDATING: 'validating',
  TESTING: 'testing',
  PREDICTING: 'predicting',
});

const REDUCE_FNS = Object.freeze({
  mean: acc => acc.count === 0 ? 0 : acc.sum / acc.count,
  sum: acc => acc.sum,
  min: acc => acc.min,
  max: acc => acc.max,
  last: acc => acc.last,
});

function createAccumulator() {
  return { sum: 0, count: 0, min: Infinity, max: -Infinity, last: 0 };
}

function updateAccumulator(acc, value) {
  const v = typeof value === 'number' ? value : value.item();
  acc.sum += v;
  acc.count += 1;
  if (v < acc.min) acc.min = v;
  if (v > acc.max) acc.max = v;
  acc.last = v;
}

export class MetricAccumulator {
  constructor() {
    this._accumulators = new Map();
    this._reduceFns = new Map();
  }

  update(name, value, reduceFx = 'mean') {
    if (!this._accumulators.has(name)) {
      this._accumulators.set(name, createAccumulator());
      this._reduceFns.set(name, reduceFx);
    }
    updateAccumulator(this._accumulators.get(name), value);
  }

  compute(name) {
    const acc = this._accumulators.get(name);
    if (!acc) return undefined;
    const fn = REDUCE_FNS[this._reduceFns.get(name)];
    return fn(acc);
  }

  computeAll() {
    const result = {};
    for (const [name] of this._accumulators) {
      result[name] = this.compute(name);
    }
    return result;
  }

  reset() {
    this._accumulators.clear();
    this._reduceFns.clear();
  }

  has(name) {
    return this._accumulators.has(name);
  }

  get size() {
    return this._accumulators.size;
  }
}

export class TrainerState {
  constructor() {
    this.stage = Stage.IDLE;
    this.epoch = 0;
    this.globalStep = 0;
    this.maxEpochs = 0;
    this.maxSteps = -1;
    this.shouldStop = false;
    this.stepMetrics = new MetricAccumulator();
    this.epochMetrics = new MetricAccumulator();
  }

  resetEpochMetrics() {
    this.epochMetrics.reset();
  }

  resetStepMetrics() {
    this.stepMetrics.reset();
  }
}

export class SingleDeviceStrategy {
  constructor() {
    this.device = null;
  }

  setup(model, device) {
    this.device = device;
    if ((device.type === 'gpu' || device.type === 'webgpu') && typeof model.to === 'function') model.to(device);
  }

  toDevice(value) {
    if (!this.device || (this.device.type !== 'gpu' && this.device.type !== 'webgpu')) return value;
    if (value && value.device && typeof value.to === 'function') return value.to(this.device);
    if (Array.isArray(value)) {
      const moved = new Array(value.length);
      for (let i = 0; i < value.length; i++) moved[i] = this.toDevice(value[i]);
      return moved;
    }
    return value;
  }

  backward(loss) {
    loss.backward();
  }

  optimizerStep(optimizer) {
    optimizer.step();
  }
}
