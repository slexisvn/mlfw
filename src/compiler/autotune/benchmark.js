import { BackendPipeline } from '../../backend/pipeline.js';

const MAX_MEASURED_SERIAL_TRIPS = 1e6;

function maxSerialTripCount(node, acc) {
  if (!node) return 0;
  if (node.type === 'ForNode') {
    const ext = node.extent && node.extent.type === 'IntImmNode' ? node.extent.value : 1;
    const next = node.threadTag ? acc : acc * ext;
    return maxSerialTripCount(node.body, next);
  }
  if (node.type === 'BlockNode') {
    return Math.max(acc, maxSerialTripCount(node.body, acc), node.initBody ? maxSerialTripCount(node.initBody, acc) : 0);
  }
  if (node.type === 'SeqNode') {
    let m = acc;
    for (const s of node.stmts) m = Math.max(m, maxSerialTripCount(s, acc));
    return m;
  }
  if (node.type === 'IfThenElseNode') {
    return Math.max(maxSerialTripCount(node.thenBody, acc), node.elseBody ? maxSerialTripCount(node.elseBody, acc) : acc);
  }
  if (node.type === 'AllocateNode' || node.type === 'LetStmtNode') return maxSerialTripCount(node.body, acc);
  return acc;
}

export class BenchmarkResult {
  constructor(medianMs, minMs, samples, totalBytes, trimmedMeanMs = null, cv = 0) {
    this.medianMs = medianMs;
    this.minMs = minMs;
    this.trimmedMeanMs = trimmedMeanMs == null ? medianMs : trimmedMeanMs;
    this.cv = cv;
    this.samples = samples;
    this._totalBytes = totalBytes;
  }

  get throughputGBs() {
    if (this.minMs <= 0 || !this._totalBytes) return 0;
    return this._totalBytes / (this.minMs * 1e6);
  }
}

export function robustStats(samples, trimFraction = 0.1) {
  const sorted = samples.slice().sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return { median: 0, min: 0, trimmedMean: 0, cv: 0 };

  const median = sorted[n >> 1];
  const min = sorted[0];

  const trim = Math.floor(n * trimFraction);
  const lo = trim;
  const hi = n - trim > lo ? n - trim : n;

  let sum = 0;
  let count = 0;
  for (let i = lo; i < hi; i++) {
    sum += sorted[i];
    count++;
  }
  const trimmedMean = count > 0 ? sum / count : median;

  let varSum = 0;
  for (let i = lo; i < hi; i++) {
    const d = sorted[i] - trimmedMean;
    varSum += d * d;
  }
  const std = count > 1 ? Math.sqrt(varSum / (count - 1)) : 0;
  const cv = trimmedMean > 0 ? std / trimmedMean : 0;

  return { median, min, trimmedMean, cv };
}

export class BenchmarkRunner {
  constructor(target, config = {}) {
    this.target = target;
    this.warmup = config.warmup ?? 3;
    this.repeat = config.repeat ?? 10;
    this.minRepeatMs = config.minRepeatMs ?? 0;
    this.maxCv = config.maxCv ?? 0;
    this.maxReMeasures = config.maxReMeasures ?? 1;
    this.measurer = config.measurer || null;
    this._bufferCache = new Map();
  }

  _getOrAllocBuffers(primFunc) {
    let totalBytes = 0;
    const sizes = [];
    for (const [, buf] of primFunc.bufferMap) {
      const numel = Math.max(buf.numel(), 1);
      sizes.push(numel);
      const bytes = buf.sizeInBytes();
      if (bytes > 0) totalBytes += bytes;
    }

    const cacheKey = sizes.join(',');
    let buffers = this._bufferCache.get(cacheKey);
    if (!buffers) {
      buffers = sizes.map(n => new Float32Array(n));
      this._bufferCache.set(cacheKey, buffers);
    }
    for (const buf of buffers) {
      for (let i = 0; i < buf.length; i++) buf[i] = Math.random() * 2 - 1;
    }
    return { buffers, totalBytes };
  }

  run(primFunc) {
    if (!primFunc || !primFunc.body) return null;
    if (!this.target.isCPU()) {
      return this.measurer ? this._runMeasured(primFunc) : null;
    }

    const backend = new BackendPipeline(this.target);
    let compiled;
    try {
      compiled = backend.compile(primFunc);
    } catch {
      return null;
    }

    if (compiled.metadata.kind !== 'js') return null;

    let fn;
    try {
      fn = new Function('return ' + compiled.source)();
    } catch {
      return null;
    }

    const { buffers, totalBytes } = this._getOrAllocBuffers(primFunc);

    for (let i = 0; i < this.warmup; i++) {
      try { fn(...buffers); } catch { return null; }
    }

    const samples = [];
    let stats = null;
    for (let round = 0; round <= this.maxReMeasures; round++) {
      this._collect(fn, buffers, samples);
      stats = robustStats(samples);
      if (this.maxCv <= 0 || stats.cv <= this.maxCv) break;
    }

    return new BenchmarkResult(stats.median, stats.min, samples, totalBytes, stats.trimmedMean, stats.cv);
  }

  _runMeasured(primFunc) {
    if (primFunc.shapeParams && primFunc.shapeParams.length > 0) return null;
    if (maxSerialTripCount(primFunc.body, 1) > MAX_MEASURED_SERIAL_TRIPS) return null;
    let compiled;
    try {
      compiled = new BackendPipeline(this.target).compile(primFunc);
    } catch {
      return null;
    }
    const byteSizes = [];
    let totalBytes = 0;
    for (const [, buf] of primFunc.bufferMap) {
      const bytes = Math.max(buf.sizeInBytes(), 1);
      byteSizes.push(bytes);
      totalBytes += bytes;
    }
    let samples;
    try {
      samples = this.measurer(compiled, byteSizes, [], { warmup: this.warmup, repeat: this.repeat });
    } catch {
      return null;
    }
    if (!samples || samples.length === 0) return null;
    const stats = robustStats(samples);
    return new BenchmarkResult(stats.median, stats.min, samples, totalBytes, stats.trimmedMean, stats.cv);
  }

  _collect(fn, buffers, samples) {
    let totalElapsed = 0;
    const maxIterations = this.repeat * 3;
    for (let i = 0; i < maxIterations && (i < this.repeat || totalElapsed < this.minRepeatMs); i++) {
      const start = performance.now();
      fn(...buffers);
      const elapsed = performance.now() - start;
      samples.push(elapsed);
      totalElapsed += elapsed;
    }
  }
}
