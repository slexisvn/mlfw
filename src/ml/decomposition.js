import { svd } from '../tensor/ops/linalg.js';
import { matmul, add, sub, mean } from '../tensor/ops/ops.js';

export class PCA {
  constructor({ nComponents = null } = {}) {
    this.nComponents = nComponents;
    this.components_ = null;
    this.mean_ = null;
    this.explainedVariance_ = null;
    this.explainedVarianceRatio_ = null;
  }

  fit(X) {
    const n = X.shape[0];
    const d = X.shape[1];
    this.mean_ = mean(X, [0], true);
    const Xc = sub(X, this.mean_);
    const { S, V } = svd(Xc);
    const k = V.shape[1];
    const nc = Math.min(this.nComponents ?? Math.min(n, d), k);
    this.components_ = V.narrow(1, 0, nc);
    this._nc = nc;

    const svals = S.toArray();
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

  transform(X) {
    return matmul(sub(X, this.mean_), this.components_);
  }

  fit_transform(X) {
    return this.fit(X).transform(X);
  }

  inverse_transform(Xr) {
    return add(matmul(Xr, this.components_.transpose(0, 1)), this.mean_);
  }
}
