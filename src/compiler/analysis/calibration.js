import { QuantizationParams, QuantizationScheme } from '../ir/graph/quantization_types.js';
import { TensorType, isFloatType } from '../ir/graph/types.js';

export class ValueObserver {
  constructor() {
    this.min = Infinity;
    this.max = -Infinity;
    this.count = 0;
    this.histogram = null;
  }

  update(data) {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < this.min) this.min = v;
      if (v > this.max) this.max = v;
    }
    this.count += data.length;
    if (this.histogram) {
      this.histogram.update(data);
    }
  }

  enableHistogram(numBins = 2048) {
    this.histogram = new HistogramCollector(numBins);
  }
}

export class HistogramCollector {
  constructor(numBins = 2048) {
    this.numBins = numBins;
    this.bins = new Float64Array(numBins);
    this.rangeMin = 0;
    this.rangeMax = 0;
    this.initialized = false;
    this.totalCount = 0;
  }

  update(data) {
    if (!this.initialized) {
      this._initRange(data);
    }

    const range = this.rangeMax - this.rangeMin;
    if (range <= 0) return;
    const scale = this.numBins / range;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < this.rangeMin || v > this.rangeMax) {
        this._expandAndRebucket(data);
        return;
      }
      const idx = Math.min(this.numBins - 1, Math.floor((v - this.rangeMin) * scale));
      this.bins[idx]++;
    }
    this.totalCount += data.length;
  }

  _initRange(data) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    if (min === max) max = min + 1;
    this.rangeMin = min;
    this.rangeMax = max;
    this.initialized = true;
  }

  _expandAndRebucket(newData) {
    let newMin = this.rangeMin, newMax = this.rangeMax;
    for (let i = 0; i < newData.length; i++) {
      if (newData[i] < newMin) newMin = newData[i];
      if (newData[i] > newMax) newMax = newData[i];
    }

    const oldBins = new Float64Array(this.bins);
    const oldMin = this.rangeMin;
    const oldMax = this.rangeMax;
    const oldRange = oldMax - oldMin;

    this.rangeMin = newMin;
    this.rangeMax = newMax;
    const newRange = newMax - newMin;
    const newScale = this.numBins / newRange;

    this.bins.fill(0);

    if (oldRange > 0) {
      const oldBinWidth = oldRange / this.numBins;
      for (let i = 0; i < this.numBins; i++) {
        if (oldBins[i] === 0) continue;
        const center = oldMin + (i + 0.5) * oldBinWidth;
        const newIdx = Math.min(this.numBins - 1, Math.floor((center - newMin) * newScale));
        this.bins[newIdx] += oldBins[i];
      }
    }

    for (let i = 0; i < newData.length; i++) {
      const idx = Math.min(this.numBins - 1, Math.floor((newData[i] - newMin) * newScale));
      this.bins[idx]++;
    }
    this.totalCount += newData.length;
  }

  computePercentileThreshold(percentile) {
    const target = this.totalCount * percentile;
    let cumulative = 0;
    const binWidth = (this.rangeMax - this.rangeMin) / this.numBins;
    for (let i = 0; i < this.numBins; i++) {
      cumulative += this.bins[i];
      if (cumulative >= target) {
        return this.rangeMin + (i + 1) * binWidth;
      }
    }
    return this.rangeMax;
  }

  computeEntropyThreshold(numQuantBins) {
    if (this.totalCount === 0) return this.rangeMax;

    const ref = new Float64Array(this.numBins);
    for (let i = 0; i < this.numBins; i++) {
      ref[i] = this.bins[i] / this.totalCount;
    }

    let bestThresholdIdx = this.numBins;
    let bestKL = Infinity;

    for (let t = numQuantBins; t <= this.numBins; t++) {
      const quantBinWidth = t / numQuantBins;
      const q = new Float64Array(t);

      for (let i = 0; i < numQuantBins; i++) {
        const start = Math.floor(i * quantBinWidth);
        const end = Math.min(t, Math.floor((i + 1) * quantBinWidth));
        let sum = 0;
        for (let j = start; j < end; j++) sum += ref[j];
        const binCount = end - start;
        if (binCount > 0) {
          const avg = sum / binCount;
          for (let j = start; j < end; j++) q[j] = avg;
        }
      }

      let kl = 0;
      for (let i = 0; i < t; i++) {
        if (ref[i] > 0 && q[i] > 0) {
          kl += ref[i] * Math.log(ref[i] / q[i]);
        }
      }

      if (kl < bestKL) {
        bestKL = kl;
        bestThresholdIdx = t;
      }
    }

    const binWidth = (this.rangeMax - this.rangeMin) / this.numBins;
    return this.rangeMin + bestThresholdIdx * binWidth;
  }
}

export class CalibrationCollector {
  constructor(mode = 'minmax') {
    this.observers = new Map();
    this.mode = mode;
  }

  attach(func) {
    for (const op of func.ops()) {
      for (let i = 0; i < op.numResults; i++) {
        const val = op.getResult(i);
        if (val.type instanceof TensorType && isFloatType(val.type.dtype)) {
          const obs = new ValueObserver();
          if (this.mode === 'entropy' || this.mode === 'percentile') {
            obs.enableHistogram();
          }
          this.observers.set(val, obs);
        }
      }
    }
    for (const arg of func.args) {
      if (arg.type instanceof TensorType && isFloatType(arg.type.dtype)) {
        const obs = new ValueObserver();
        if (this.mode === 'entropy' || this.mode === 'percentile') {
          obs.enableHistogram();
        }
        this.observers.set(arg, obs);
      }
    }
  }

  observe(value, data) {
    const obs = this.observers.get(value);
    if (obs) obs.update(data);
  }

  getResult() {
    return new CalibrationResult(this.observers, this.mode);
  }
}

export class CalibrationResult {
  constructor(observers, mode) {
    this._observers = observers;
    this._mode = mode;
  }

  getRange(value) {
    const obs = this._observers.get(value);
    if (!obs || obs.count === 0) return null;
    return { min: obs.min, max: obs.max };
  }

  getQuantParams(value, scheme, dtype) {
    const range = this.getRange(value);
    if (!range) return null;

    if (this._mode === 'percentile' && this._observers.get(value).histogram) {
      const hist = this._observers.get(value).histogram;
      const lo = -hist.computePercentileThreshold(0.999);
      const hi = hist.computePercentileThreshold(0.999);
      return QuantizationParams.fromRange(lo, hi, scheme, dtype);
    }

    if (this._mode === 'entropy' && this._observers.get(value).histogram) {
      const hist = this._observers.get(value).histogram;
      const numQuantBins = dtype === 'ui8' ? 256 : 255;
      const threshold = hist.computeEntropyThreshold(numQuantBins);
      return QuantizationParams.fromRange(-threshold, threshold, scheme, dtype);
    }

    return QuantizationParams.fromRange(range.min, range.max, scheme, dtype);
  }

  hasData(value) {
    const obs = this._observers.get(value);
    return obs && obs.count > 0;
  }

  values() {
    return this._observers.keys();
  }
}
