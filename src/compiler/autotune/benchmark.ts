import { BackendPipeline } from '../../backend/pipeline.js';
import { random } from '../../util/random.js';

import type { TirNode, PrimFunc, ForNode, BlockNode, SeqNode, IfThenElseNode, AllocateNode, LetStmtNode, IntImmNode } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';

export type RobustStats = { median: number; min: number; trimmedMean: number; cv: number };
export type MeasurerLike = (compiled: unknown, byteSizes: readonly number[], extra: readonly unknown[], opts: { warmup: number; repeat: number }) => number[];
export type CompiledKernelLike = { metadata: { kind?: string }; source: string };
export type BenchmarkWarnFn = (stage: string, subject: string | null, error: unknown) => void;
export type BenchmarkConfig = Readonly<{
  warmup?: number;
  repeat?: number;
  minRepeatMs?: number;
  maxCv?: number;
  maxReMeasures?: number;
  measurer?: MeasurerLike | null;
  warn?: BenchmarkWarnFn | null;
}>;

const MAX_MEASURED_SERIAL_TRIPS = 1e6;

function maxSerialTripCount(node: TirNode | null | undefined, acc: number): number {
  if (!node) return 0;
  if (node.type === 'ForNode') {
    const f = node as ForNode;
    const ext = f.extent && f.extent.type === 'IntImmNode' ? (f.extent as IntImmNode).value : 1;
    const next = f.threadTag ? acc : acc * ext;
    return maxSerialTripCount(f.body, next);
  }
  if (node.type === 'BlockNode') {
    const b = node as BlockNode;
    return Math.max(acc, maxSerialTripCount(b.body, acc), b.initBody ? maxSerialTripCount(b.initBody, acc) : 0);
  }
  if (node.type === 'SeqNode') {
    let m = acc;
    for (const st of (node as SeqNode).stmts) m = Math.max(m, maxSerialTripCount(st, acc));
    return m;
  }
  if (node.type === 'IfThenElseNode') {
    const ite = node as IfThenElseNode;
    return Math.max(maxSerialTripCount(ite.thenBody, acc), ite.elseBody ? maxSerialTripCount(ite.elseBody, acc) : acc);
  }
  if (node.type === 'AllocateNode' || node.type === 'LetStmtNode') return maxSerialTripCount((node as AllocateNode | LetStmtNode).body, acc);
  return acc;
}

export class BenchmarkResult {
  medianMs: number;
  minMs: number;
  trimmedMeanMs: number;
  cv: number;
  samples: number[];
  private _totalBytes: number;

  constructor(medianMs: number, minMs: number, samples: number[], totalBytes: number, trimmedMeanMs: number | null = null, cv = 0) {
    this.medianMs = medianMs;
    this.minMs = minMs;
    this.trimmedMeanMs = trimmedMeanMs == null ? medianMs : trimmedMeanMs;
    this.cv = cv;
    this.samples = samples;
    this._totalBytes = totalBytes;
  }

  get throughputGBs(): number {
    if (this.minMs <= 0 || !this._totalBytes) return 0;
    return this._totalBytes / (this.minMs * 1e6);
  }
}

export function robustStats(samples: readonly number[], trimFraction = 0.1): RobustStats {
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
  target: ScheduleTarget;
  warmup: number;
  repeat: number;
  minRepeatMs: number;
  maxCv: number;
  maxReMeasures: number;
  measurer: MeasurerLike | null;
  private _warn: BenchmarkWarnFn | null;
  private _bufferCache: Map<string, Float32Array[]>;

  constructor(target: ScheduleTarget, config: BenchmarkConfig = {}) {
    this.target = target;
    this.warmup = config.warmup ?? 3;
    this.repeat = config.repeat ?? 10;
    this.minRepeatMs = config.minRepeatMs ?? 0;
    this.maxCv = config.maxCv ?? 0;
    this.maxReMeasures = config.maxReMeasures ?? 1;
    this.measurer = config.measurer || null;
    this._warn = config.warn || null;
    this._bufferCache = new Map();
  }

  _record(stage: string, error: unknown): void {
    if (this._warn) this._warn(stage, null, error);
  }

  _getOrAllocBuffers(primFunc: PrimFunc): { buffers: Float32Array[]; totalBytes: number } {
    let totalBytes = 0;
    const sizes: number[] = [];
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
      for (let i = 0; i < buf.length; i++) buf[i] = random() * 2 - 1;
    }
    return { buffers, totalBytes };
  }

  run(primFunc: PrimFunc | null | undefined): BenchmarkResult | null {
    if (!primFunc || !primFunc.body) return null;
    if (!this.target.isCPU()) {
      return this.measurer ? this._runMeasured(primFunc) : null;
    }

    const backend = new BackendPipeline(this.target);
    let compiled;
    try {
      compiled = backend.compile(primFunc);
    } catch (e) {
      this._record('benchmark-compile', e);
      return null;
    }

    if ((compiled as CompiledKernelLike).metadata.kind !== 'js') return null;

    let fn: (...args: unknown[]) => unknown;
    try {
      fn = new Function('return ' + (compiled as CompiledKernelLike).source)() as (...args: unknown[]) => unknown;
    } catch (e) {
      this._record('benchmark-construct-fn', e);
      return null;
    }

    const { buffers, totalBytes } = this._getOrAllocBuffers(primFunc);

    for (let i = 0; i < this.warmup; i++) {
      try { fn(...buffers); } catch (e) { this._record('benchmark-warmup-run', e); return null; }
    }

    const samples: number[] = [];
    let stats: RobustStats | null = null;
    for (let round = 0; round <= this.maxReMeasures; round++) {
      this._collect(fn, buffers, samples);
      stats = robustStats(samples);
      if (this.maxCv <= 0 || stats.cv <= this.maxCv) break;
    }

    const st = stats as RobustStats;
    return new BenchmarkResult(st.median, st.min, samples, totalBytes, st.trimmedMean, st.cv);
  }

  _runMeasured(primFunc: PrimFunc): BenchmarkResult | null {
    if (primFunc.shapeParams && primFunc.shapeParams.length > 0) return null;
    if (maxSerialTripCount(primFunc.body, 1) > MAX_MEASURED_SERIAL_TRIPS) return null;
    let compiled: unknown;
    try {
      compiled = new BackendPipeline(this.target).compile(primFunc);
    } catch (e) {
      this._record('measured-compile', e);
      return null;
    }
    const byteSizes: number[] = [];
    let totalBytes = 0;
    for (const [, buf] of primFunc.bufferMap) {
      const bytes = Math.max(buf.sizeInBytes(), 1);
      byteSizes.push(bytes);
      totalBytes += bytes;
    }
    let samples: number[];
    try {
      samples = (this.measurer as MeasurerLike)(compiled, byteSizes, [], { warmup: this.warmup, repeat: this.repeat });
    } catch (e) {
      this._record('measurer', e);
      return null;
    }
    if (!samples || samples.length === 0) return null;
    const stats = robustStats(samples);
    const st = stats as RobustStats;
    return new BenchmarkResult(st.median, st.min, samples, totalBytes, st.trimmedMean, st.cv);
  }

  _collect(fn: (...args: unknown[]) => unknown, buffers: readonly Float32Array[], samples: number[]): void {
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
