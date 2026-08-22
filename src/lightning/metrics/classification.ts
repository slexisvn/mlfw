import { Metric } from './metric.js';
import { argmaxRow } from './row_ops.js';
import type { AverageMode, ClassificationTask, TensorLike } from '../types.js';

type ClassificationOptions = {
  task?: ClassificationTask;
  numClasses?: number;
  average?: AverageMode;
};

export class Precision extends Metric {
  private _task: ClassificationTask;
  private _numClasses: number;
  private _average: AverageMode;
  private _tp: Int32Array;
  private _fp: Int32Array;
  private _support: Int32Array;

  constructor({ task = 'multiclass', numClasses = 2, average = 'macro' }: ClassificationOptions = {}) {
    super();
    this._task = task;
    this._numClasses = numClasses;
    this._average = average;
    this._tp = new Int32Array(numClasses);
    this._fp = new Int32Array(numClasses);
    this._support = new Int32Array(numClasses);
  }

  update(preds: TensorLike, target: TensorLike): void {
    updateConfusionCounters(
      preds, target, this._tp, this._fp, null,
      this._support, this._task, this._numClasses
    );
  }

  compute(): number | number[] {
    return computePrecision(this._tp, this._fp, this._support, this._numClasses, this._average);
  }

  reset(): void {
    super.reset();
    this._tp.fill(0);
    this._fp.fill(0);
    this._support.fill(0);
  }
}

export class Recall extends Metric {
  private _task: ClassificationTask;
  private _numClasses: number;
  private _average: AverageMode;
  private _tp: Int32Array;
  private _fn: Int32Array;
  private _support: Int32Array;

  constructor({ task = 'multiclass', numClasses = 2, average = 'macro' }: ClassificationOptions = {}) {
    super();
    this._task = task;
    this._numClasses = numClasses;
    this._average = average;
    this._tp = new Int32Array(numClasses);
    this._fn = new Int32Array(numClasses);
    this._support = new Int32Array(numClasses);
  }

  update(preds: TensorLike, target: TensorLike): void {
    updateConfusionCounters(
      preds, target, this._tp, null, this._fn,
      this._support, this._task, this._numClasses
    );
  }

  compute(): number | number[] {
    return computeRecall(this._tp, this._fn, this._support, this._numClasses, this._average);
  }

  reset(): void {
    super.reset();
    this._tp.fill(0);
    this._fn.fill(0);
    this._support.fill(0);
  }
}

export class F1Score extends Metric {
  private _task: ClassificationTask;
  private _numClasses: number;
  private _average: AverageMode;
  private _tp: Int32Array;
  private _fp: Int32Array;
  private _fn: Int32Array;
  private _support: Int32Array;

  constructor({ task = 'multiclass', numClasses = 2, average = 'macro' }: ClassificationOptions = {}) {
    super();
    this._task = task;
    this._numClasses = numClasses;
    this._average = average;
    this._tp = new Int32Array(numClasses);
    this._fp = new Int32Array(numClasses);
    this._fn = new Int32Array(numClasses);
    this._support = new Int32Array(numClasses);
  }

  update(preds: TensorLike, target: TensorLike): void {
    updateConfusionCounters(
      preds, target, this._tp, this._fp, this._fn,
      this._support, this._task, this._numClasses
    );
  }

  compute(): number | number[] {
    const p = computePrecisionPerClass(this._tp, this._fp, this._numClasses);
    const r = computeRecallPerClass(this._tp, this._fn, this._numClasses);
    const f1 = new Float64Array(this._numClasses);
    for (let c = 0; c < this._numClasses; c++) {
      const denom = p[c] + r[c];
      f1[c] = denom > 0 ? (2 * p[c] * r[c]) / denom : 0;
    }
    return aggregate(f1, this._support, this._numClasses, this._average);
  }

  reset(): void {
    super.reset();
    this._tp.fill(0);
    this._fp.fill(0);
    this._fn.fill(0);
    this._support.fill(0);
  }
}

function updateConfusionCounters(
  preds: TensorLike,
  target: TensorLike,
  tp: Int32Array,
  fp: Int32Array | null,
  fn: Int32Array | null,
  support: Int32Array,
  task: ClassificationTask,
  numClasses: number
): void {
  const predData = preds._impl.storage.data;
  const targetData = target._impl.storage.data;
  const batchSize = targetData.length;
  const hasMultipleCols = preds.shape.length >= 2 && preds.shape[1] > 1;

  for (let i = 0; i < batchSize; i++) {
    const t = (targetData[i] as number) | 0;
    let p;

    if (task === 'binary') {
      p = predData[i] >= 0.5 ? 1 : 0;
    } else if (hasMultipleCols) {
      p = argmaxRow(predData, i, numClasses);
    } else {
      p = (predData[i] as number) | 0;
    }

    support[t]++;
    if (p === t) {
      if (tp) tp[t]++;
    } else {
      if (fp) fp[p]++;
      if (fn) fn[t]++;
    }
  }
}


function computePrecisionPerClass(tp: Int32Array, fp: Int32Array, numClasses: number): Float64Array {
  const result = new Float64Array(numClasses);
  for (let c = 0; c < numClasses; c++) {
    const denom = tp[c] + fp[c];
    result[c] = denom > 0 ? tp[c] / denom : 0;
  }
  return result;
}

function computeRecallPerClass(tp: Int32Array, fn: Int32Array, numClasses: number): Float64Array {
  const result = new Float64Array(numClasses);
  for (let c = 0; c < numClasses; c++) {
    const denom = tp[c] + fn[c];
    result[c] = denom > 0 ? tp[c] / denom : 0;
  }
  return result;
}

function computePrecision(tp: Int32Array, fp: Int32Array, support: Int32Array, numClasses: number, average: AverageMode): number | number[] {
  const perClass = computePrecisionPerClass(tp, fp, numClasses);
  return aggregate(perClass, support, numClasses, average);
}

function computeRecall(tp: Int32Array, fn: Int32Array, support: Int32Array, numClasses: number, average: AverageMode): number | number[] {
  const perClass = computeRecallPerClass(tp, fn, numClasses);
  return aggregate(perClass, support, numClasses, average);
}

function aggregate(perClass: Float64Array, support: Int32Array, numClasses: number, average: AverageMode): number | number[] {
  if (average === 'none') return [...perClass];

  if (average === 'micro') {
    let totalCorrect = 0;
    let totalSupport = 0;
    for (let c = 0; c < numClasses; c++) {
      totalCorrect += perClass[c] * support[c];
      totalSupport += support[c];
    }
    return totalSupport > 0 ? totalCorrect / totalSupport : 0;
  }

  if (average === 'weighted') {
    let totalSupport = 0;
    let weightedSum = 0;
    for (let c = 0; c < numClasses; c++) {
      weightedSum += perClass[c] * support[c];
      totalSupport += support[c];
    }
    return totalSupport > 0 ? weightedSum / totalSupport : 0;
  }

  let activeClasses = 0;
  let total = 0;
  for (let c = 0; c < numClasses; c++) {
    if (support[c] > 0) {
      total += perClass[c];
      activeClasses++;
    }
  }
  return activeClasses > 0 ? total / activeClasses : 0;
}
