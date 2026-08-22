import { Estimator } from './estimator.js';
import { _dispatch } from '../tensor/ops/ops.js';
import './metrics.js';
import type { MLTensor } from './types.js';

export class GaussianNB extends Estimator {
  protected readonly _classify = true;
  means_: MLTensor | null;
  variances_: MLTensor | null;
  priors_: MLTensor | null;
  classes_: MLTensor | null;

  constructor() {
    super();
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

}
