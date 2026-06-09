import { Metric } from './metric.js';
import { argmax, eq, sum, gt } from '../../tensor/ops/ops.js';

export class Accuracy extends Metric {
  constructor({ task = 'multiclass', numClasses = null, topK = 1, threshold = 0.5 } = {}) {
    super();
    this._task = task;
    this._numClasses = numClasses;
    this._topK = topK;
    this._threshold = threshold;
    this._correct = 0;
    this._total = 0;
  }

  update(preds, target) {
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

  compute() {
    return this._total === 0 ? 0 : this._correct / this._total;
  }

  reset() {
    super.reset();
    this._correct = 0;
    this._total = 0;
  }

  _updateBinary(predData, targetData) {
    const n = targetData.length;
    for (let i = 0; i < n; i++) {
      const pred = predData[i] >= this._threshold ? 1 : 0;
      if (pred === targetData[i]) this._correct++;
    }
    this._total += n;
  }

  _updateMulticlass(preds, target) {
    const shape = preds.shape;
    if (shape.length < 2) {
      const predData = preds._impl.storage.data;
      const targetData = target._impl.storage.data;
      const n = targetData.length;
      for (let i = 0; i < n; i++) {
        if (Math.round(predData[i]) === targetData[i]) this._correct++;
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

  _updateMultilabel(predData, targetData) {
    const n = targetData.length;
    for (let i = 0; i < n; i++) {
      const pred = predData[i] >= this._threshold ? 1 : 0;
      if (pred === targetData[i]) this._correct++;
    }
    this._total += n;
  }
}

function topKIndices(data, offset, length, k) {
  const heap = [];
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

function heapify(heap) {
  for (let i = (heap.length >>> 1) - 1; i >= 0; i--) siftDown(heap, i);
}

function siftDown(heap, i) {
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
