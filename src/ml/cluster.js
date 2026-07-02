import { _dispatch } from '../tensor/ops/ops.js';

export class KMeans {
  constructor({ nClusters = 8, maxIter = 300, nInit = 10, randomState = 0 } = {}) {
    this.nClusters = nClusters;
    this.maxIter = maxIter;
    this.nInit = nInit;
    this.randomState = randomState;
    this.clusterCenters_ = null;
    this.labels_ = null;
    this.inertia_ = null;
  }

  fit(X) {
    const [centers, labels, inertia] = _dispatch('kmeans', X, this.nClusters, this.maxIter, this.nInit, this.randomState);
    this.clusterCenters_ = centers;
    this.labels_ = labels;
    this.inertia_ = inertia.item();
    return this;
  }

  predict(X) {
    return _dispatch('kmeans_predict', X, this.clusterCenters_);
  }

  fit_predict(X) {
    this.fit(X);
    return this.labels_;
  }
}
