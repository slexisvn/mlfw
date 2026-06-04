import { BackendPipeline } from '../backend/pipeline.js';

export class BenchmarkResult {
  constructor(medianMs, minMs, samples, totalBytes) {
    this.medianMs = medianMs;
    this.minMs = minMs;
    this.samples = samples;
    this._totalBytes = totalBytes;
  }

  get throughputGBs() {
    if (this.minMs <= 0 || !this._totalBytes) return 0;
    return this._totalBytes / (this.minMs * 1e6);
  }
}

export class BenchmarkRunner {
  constructor(target, config = {}) {
    this.target = target;
    this.warmup = config.warmup ?? 3;
    this.repeat = config.repeat ?? 10;
    this.minRepeatMs = config.minRepeatMs ?? 0;
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
      for (const buf of buffers) {
        for (let i = 0; i < buf.length; i++) buf[i] = Math.random() * 2 - 1;
      }
      this._bufferCache.set(cacheKey, buffers);
    }
    return { buffers, totalBytes };
  }

  run(primFunc) {
    if (!this.target.isCPU()) return null;
    if (!primFunc || !primFunc.body) return null;

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
    let totalElapsed = 0;
    const maxIterations = this.repeat * 3;

    for (let i = 0; i < this.repeat || totalElapsed < this.minRepeatMs; i++) {
      if (i >= maxIterations) break;
      const start = performance.now();
      fn(...buffers);
      const elapsed = performance.now() - start;
      samples.push(elapsed);
      totalElapsed += elapsed;
    }

    samples.sort((a, b) => a - b);

    return new BenchmarkResult(
      samples[samples.length >> 1],
      samples[0],
      samples,
      totalBytes
    );
  }
}
