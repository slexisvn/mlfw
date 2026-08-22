import { Metric } from './metric.js';
import type { NumericTypedArray } from '../../tensor/types/dtype.js';
import type { AccuracyTask, TensorLike } from '../types.js';

type AccuracyOptions = {
  task?: AccuracyTask;
  numClasses?: number | null;
  topK?: number;
  threshold?: number;
};

export class Accuracy extends Metric {
  private _task: AccuracyTask;
  private _topK: number;
  private _threshold: number;
  private _correct: number;
  private _total: number;

  constructor({ task = 'multiclass', numClasses = null, topK = 1, threshold = 0.5 }: AccuracyOptions = {}) {
    super();
    this._task = task;
    this._topK = topK;
    this._threshold = threshold;
    this._correct = 0;
    this._total = 0;
  }

  update(preds: TensorLike, target: TensorLike): void {
    const predData = preds._impl.storage.data;
    const targetData = target._impl.storage.data;

    if (this._task === 'binary') {
      this._updateBinary(predData, targetData);
    } else if (this._task === 'multiclass') {
      this._updateMulticlass(preds, target);
    } else if (this._task === 'multilabel') {
      this._updateMultilabel(predData, targetData);
    }
  }

  compute(): number {
    return this._total === 0 ? 0 : this._correct / this._total;
  }

  reset(): void {
    super.reset();
    this._correct = 0;
    this._total = 0;
  }

  private _updateBinary(predData: NumericTypedArray, targetData: NumericTypedArray): void {
    const n = targetData.length;
    for (let i = 0; i < n; i++) {
      const pred = predData[i] >= this._threshold ? 1 : 0;
      if (pred === targetData[i]) this._correct++;
    }
    this._total += n;
  }

  private _updateMulticlass(preds: TensorLike, target: TensorLike): void {
    const shape = preds.shape;
    if (shape.length < 2) {
      const predData = preds._impl.storage.data;
      const targetData = target._impl.storage.data;
      const n = targetData.length;
      for (let i = 0; i < n; i++) {
        if (Math.round(predData[i] as number) === targetData[i]) this._correct++;
      }
      this._total += n;
      return;
    }

    const batchSize = shape[0];
    const numClasses = shape[1];
    const predData = preds._impl.storage.data;
    const targetData = target._impl.storage.data;

    if (this._topK === 1) {
      for (let i = 0; i < batchSize; i++) {
        let maxIdx = 0;
        let maxVal = predData[i * numClasses];
        for (let j = 1; j < numClasses; j++) {
          const val = predData[i * numClasses + j];
          if (val > maxVal) { maxVal = val; maxIdx = j; }
        }
        if (maxIdx === targetData[i]) this._correct++;
      }
    } else {
      for (let i = 0; i < batchSize; i++) {
        const indices = topKIndices(predData, i * numClasses, numClasses, this._topK);
        for (let k = 0; k < indices.length; k++) {
          if (indices[k] === targetData[i]) { this._correct++; break; }
        }
      }
    }
    this._total += batchSize;
  }

  private _updateMultilabel(predData: NumericTypedArray, targetData: NumericTypedArray): void {
    const n = targetData.length;
    for (let i = 0; i < n; i++) {
      const pred = predData[i] >= this._threshold ? 1 : 0;
      if (pred === targetData[i]) this._correct++;
    }
    this._total += n;
  }
}

type HeapEntry = { val: number | bigint; idx: number };

function topKIndices(data: NumericTypedArray, offset: number, length: number, k: number): number[] {
  const heap: HeapEntry[] = [];
  for (let i = 0; i < length; i++) {
    const val = data[offset + i];
    if (heap.length < k) {
      heap.push({ val, idx: i });
      if (heap.length === k) heapify(heap);
    } else if (val > heap[0].val) {
      heap[0] = { val, idx: i };
      siftDown(heap, 0);
    }
  }
  return heap.map(h => h.idx);
}

function heapify(heap: HeapEntry[]): void {
  for (let i = (heap.length >>> 1) - 1; i >= 0; i--) siftDown(heap, i);
}

function siftDown(heap: HeapEntry[], i: number): void {
  const n = heap.length;
  while (true) {
    let smallest = i;
    const l = 2 * i + 1;
    const r = 2 * i + 2;
    if (l < n && heap[l].val < heap[smallest].val) smallest = l;
    if (r < n && heap[r].val < heap[smallest].val) smallest = r;
    if (smallest === i) break;
    const tmp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = tmp;
    i = smallest;
  }
}
