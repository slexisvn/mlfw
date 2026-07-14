import { _dispatch } from '../tensor/ops/ops.js';
import { accuracy_score, r2_score } from './metrics.js';
import type { MLTensor } from './types.js';

class BaseKNN {
  nNeighbors: number;
  protected _classify: boolean;
  protected _X: MLTensor | null;
  protected _y: MLTensor | null;

  constructor(nNeighbors: number, classify: boolean) {
    this.nNeighbors = nNeighbors;
    this._classify = classify;
    this._X = null;
    this._y = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    this._X = X;
    this._y = y;
    return this;
  }

  predict(X: MLTensor): MLTensor {
    return _dispatch('knn_predict', this._X, this._y, X, this.nNeighbors, this._classify) as MLTensor;
  }
}

export class KNeighborsClassifier extends BaseKNN {
  constructor({ nNeighbors = 5 } = {}) {
    super(nNeighbors, true);
  }

  score(X: MLTensor, y: MLTensor): number {
    return accuracy_score(y, this.predict(X));
  }
}

export class KNeighborsRegressor extends BaseKNN {
  constructor({ nNeighbors = 5 } = {}) {
    super(nNeighbors, false);
  }

  score(X: MLTensor, y: MLTensor): number {
    return r2_score(y, this.predict(X));
  }
}
