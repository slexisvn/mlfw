import { accuracy_score, r2_score } from './metrics.js';
import type { MLTensor } from './types.js';

export abstract class Estimator {
  protected abstract readonly _classify: boolean;

  abstract predict(X: MLTensor): MLTensor;

  score(X: MLTensor, y: MLTensor): number {
    const prediction = this.predict(X);
    return this._classify ? accuracy_score(y, prediction) : r2_score(y, prediction);
  }
}

export abstract class TransformerMixin {
  abstract fit(X: MLTensor): this;

  abstract transform(X: MLTensor): MLTensor;

  fit_transform(X: MLTensor): MLTensor {
    return this.fit(X).transform(X);
  }
}
