import { Metric } from './metric.js';

export class ConfusionMatrix extends Metric {
  constructor({ numClasses }) {
    super();
    this._numClasses = numClasses;
    this._matrix = new Int32Array(numClasses * numClasses);
  }

  update(preds, target) {
    const predData = preds._impl.storage.data;
    const targetData = target._impl.storage.data;
    const batchSize = targetData.length;
    const nc = this._numClasses;
    const hasMultipleCols = preds.shape.length >= 2 && preds.shape[1] > 1;

    for (let i = 0; i < batchSize; i++) {
      const t = targetData[i] | 0;
      let p;
      if (hasMultipleCols) {
        p = argmaxRow(predData, i, nc);
      } else {
        p = predData[i] | 0;
      }
      this._matrix[t * nc + p]++;
    }
  }

  compute() {
    const nc = this._numClasses;
    const result = [];
    for (let i = 0; i < nc; i++) {
      const row = new Array(nc);
      for (let j = 0; j < nc; j++) {
        row[j] = this._matrix[i * nc + j];
      }
      result.push(row);
    }
    return result;
  }

  reset() {
    super.reset();
    this._matrix.fill(0);
  }
}

function argmaxRow(data, row, cols) {
  let best = 0;
  let bestVal = data[row * cols];
  for (let j = 1; j < cols; j++) {
    const v = data[row * cols + j];
    if (v > bestVal) { bestVal = v; best = j; }
  }
  return best;
}
