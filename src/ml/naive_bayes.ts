import { _dispatch } from '../tensor/ops/ops.js';
import { accuracy_score } from './metrics.js';
import type { MLTensor } from './types.js';

export class GaussianNB {
  means_: MLTensor | null;
  variances_: MLTensor | null;
  priors_: MLTensor | null;
  classes_: MLTensor | null;

  constructor() {
    this.means_ = null;
    this.variances_ = null;
    this.priors_ = null;
    this.classes_ = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    const [means, variances, priors, classes] = _dispatch('gaussian_nb_fit', X, y) as [MLTensor, MLTensor, MLTensor, MLTensor];
    this.means_ = means;
    this.variances_ = variances;
    this.priors_ = priors;
    this.classes_ = classes;
    return this;
  }

  predict(X: MLTensor): MLTensor {
    return _dispatch('gaussian_nb_predict', X, this.means_, this.variances_, this.priors_, this.classes_) as MLTensor;
  }

  score(X: MLTensor, y: MLTensor): number {
    return accuracy_score(y, this.predict(X));
  }
}
