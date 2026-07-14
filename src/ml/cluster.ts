import { _dispatch } from '../tensor/ops/ops.js';
import type { MLTensor } from './types.js';

export class KMeans {
  nClusters: number;
  maxIter: number;
  nInit: number;
  randomState: number;
  clusterCenters_: MLTensor | null;
  labels_: MLTensor | null;
  inertia_: number | null;

  constructor({ nClusters = 8, maxIter = 300, nInit = 10, randomState = 0 }: { nClusters?: number; maxIter?: number; nInit?: number; randomState?: number } = {}) {
    this.nClusters = nClusters;
    this.maxIter = maxIter;
    this.nInit = nInit;
    this.randomState = randomState;
    this.clusterCenters_ = null;
    this.labels_ = null;
    this.inertia_ = null;
  }

  fit(X: MLTensor): this {
    const [centers, labels, inertia] = _dispatch('kmeans', X, this.nClusters, this.maxIter, this.nInit, this.randomState) as [MLTensor, MLTensor, MLTensor];
    this.clusterCenters_ = centers;
    this.labels_ = labels;
    this.inertia_ = Number(inertia.item());
    return this;
  }

  predict(X: MLTensor): MLTensor {
    return _dispatch('kmeans_predict', X, this.clusterCenters_) as MLTensor;
  }

  fit_predict(X: MLTensor): MLTensor | null {
    this.fit(X);
    return this.labels_;
  }
}
