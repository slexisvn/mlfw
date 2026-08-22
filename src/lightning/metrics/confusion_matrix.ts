import { Metric } from './metric.js';
import { argmaxRow } from './row_ops.js';
import type { TensorLike } from '../types.js';

type ConfusionMatrixOptions = {
  numClasses: number;
};

export class ConfusionMatrix extends Metric {
  private _numClasses: number;
  private _matrix: Int32Array;

  constructor({ numClasses }: ConfusionMatrixOptions) {
    super();
    this._numClasses = numClasses;
    this._matrix = new Int32Array(numClasses * numClasses);
  }

  update(preds: TensorLike, target: TensorLike): void {
    const predData = preds._impl.storage.data;
    const targetData = target._impl.storage.data;
    const batchSize = targetData.length;
    const nc = this._numClasses;
    const hasMultipleCols = preds.shape.length >= 2 && preds.shape[1] > 1;

    for (let i = 0; i < batchSize; i++) {
      const t = (targetData[i] as number) | 0;
      let p;
      if (hasMultipleCols) {
        p = argmaxRow(predData, i, nc);
      } else {
        p = (predData[i] as number) | 0;
      }
      this._matrix[t * nc + p]++;
    }
  }

  compute(): number[][] {
    const nc = this._numClasses;
    const result: number[][] = [];
    for (let i = 0; i < nc; i++) {
      const row = new Array(nc);
      for (let j = 0; j < nc; j++) {
        row[j] = this._matrix[i * nc + j];
      }
      result.push(row);
    }
    return result;
  }

  reset(): void {
    super.reset();
    this._matrix.fill(0);
  }
}

