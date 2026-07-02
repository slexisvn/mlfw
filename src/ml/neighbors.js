import { _dispatch } from '../tensor/ops/ops.js';
import { accuracy_score, r2_score } from './metrics.js';

class BaseKNN {
  constructor(nNeighbors, classify) {
    this.nNeighbors = nNeighbors;
    this._classify = classify;
    this._X = null;
    this._y = null;
  }

  fit(X, y) {
    this._X = X;
    this._y = y;
    return this;
  }

  predict(X) {
    return _dispatch('knn_predict', this._X, this._y, X, this.nNeighbors, this._classify);
  }
}

export class KNeighborsClassifier extends BaseKNN {
  constructor({ nNeighbors = 5 } = {}) {
    super(nNeighbors, true);
  }

  score(X, y) {
    return accuracy_score(y, this.predict(X));
  }
}

export class KNeighborsRegressor extends BaseKNN {
  constructor({ nNeighbors = 5 } = {}) {
    super(nNeighbors, false);
  }

  score(X, y) {
    return r2_score(y, this.predict(X));
  }
}
