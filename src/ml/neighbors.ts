import { Estimator } from './estimator.js';
import { _dispatch } from '../tensor/ops/ops.js';
import type { MLTensor } from './types.js';

class BaseKNN extends Estimator {
  nNeighbors: number;
  protected _classify: boolean;
  protected _X: MLTensor | null;
  protected _y: MLTensor | null;

  constructor(nNeighbors: number, classify: boolean) {
    super();
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

}

export class KNeighborsRegressor extends BaseKNN {
  constructor({ nNeighbors = 5 } = {}) {
    super(nNeighbors, false);
  }

}
