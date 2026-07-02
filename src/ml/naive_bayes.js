import { _dispatch } from '../tensor/ops/ops.js';
import { accuracy_score } from './metrics.js';

export class GaussianNB {
  constructor() {
    this.means_ = null;
    this.variances_ = null;
    this.priors_ = null;
    this.classes_ = null;
  }

  fit(X, y) {
    const [means, variances, priors, classes] = _dispatch('gaussian_nb_fit', X, y);
    this.means_ = means;
    this.variances_ = variances;
    this.priors_ = priors;
    this.classes_ = classes;
    return this;
  }

  predict(X) {
    return _dispatch('gaussian_nb_predict', X, this.means_, this.variances_, this.priors_, this.classes_);
  }

  score(X, y) {
    return accuracy_score(y, this.predict(X));
  }
}
