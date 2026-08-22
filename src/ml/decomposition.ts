import { TransformerMixin } from './estimator.js';
import { svd } from '../tensor/ops/linalg.js';
import { matmul, add, sub, mean } from '../tensor/ops/ops.js';
import type { MLTensor } from './types.js';

export class PCA extends TransformerMixin {
  nComponents: number | null;
  components_: MLTensor | null;
  mean_: MLTensor | null;
  explainedVariance_: number[] | null;
  explainedVarianceRatio_: number[] | null;

  constructor({ nComponents = null }: { nComponents?: number | null } = {}) {
    super();
    this.nComponents = nComponents;
    this.components_ = null;
    this.mean_ = null;
    this.explainedVariance_ = null;
    this.explainedVarianceRatio_ = null;
  }

  fit(X: MLTensor): this {
    const n = X.shape[0];
    const d = X.shape[1];
    this.mean_ = mean(X, [0], true) as MLTensor;
    const Xc = sub(X, this.mean_) as MLTensor;
    const { S, V } = svd(Xc) as unknown as { S: MLTensor; V: MLTensor };
    const k = V.shape[1];
    const nc = Math.min(this.nComponents ?? Math.min(n, d), k);
    this.components_ = V.narrow(1, 0, nc);

    const svals = S.toArray() as ArrayLike<number>;
    const denom = n > 1 ? n - 1 : 1;
    let total = 0;
    for (let c = 0; c < svals.length; c++) total += (svals[c] * svals[c]) / denom;
    this.explainedVariance_ = new Array(nc);
    this.explainedVarianceRatio_ = new Array(nc);
    for (let c = 0; c < nc; c++) {
      const ev = (svals[c] * svals[c]) / denom;
      this.explainedVariance_[c] = ev;
      this.explainedVarianceRatio_[c] = total > 0 ? ev / total : 0;
    }
    return this;
  }

  transform(X: MLTensor): MLTensor {
    return matmul(sub(X, this.mean_!) as MLTensor, this.components_!) as MLTensor;
  }

  inverse_transform(Xr: MLTensor): MLTensor {
    return add(matmul(Xr, this.components_!.transpose(0, 1)), this.mean_!) as MLTensor;
  }
}
