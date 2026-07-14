import { solve, lstsq } from '../tensor/ops/linalg.js';
import { matmul, add, sub, mul, div, exp, sum, mean, max, argmax, cat, _dispatch } from '../tensor/ops/ops.js';
import { ones, eye, zeros } from '../tensor/factory/creation_ops.js';
import { tensor } from '../tensor/factory/from_ops.js';
import { vectorOf, encodeLabels } from './_util.js';
import { r2_score, accuracy_score } from './metrics.js';
import type { MLTensor } from './types.js';

function addBias(X: MLTensor, fitIntercept: boolean): MLTensor {
  if (!fitIntercept) return X;
  return cat([X, ones([X.shape[0], 1], { device: X.device, dtype: X.dtype })], 1) as MLTensor;
}

function asColumn(y: MLTensor): MLTensor {
  return y.ndim === 1 ? y.reshape([y.shape[0], 1]) : y;
}

export class LinearRegression {
  fitIntercept: boolean;
  weight_: MLTensor | null;

  constructor({ fitIntercept = true }: { fitIntercept?: boolean } = {}) {
    this.fitIntercept = fitIntercept;
    this.weight_ = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    const D = addBias(X, this.fitIntercept);
    this.weight_ = lstsq(D, asColumn(y)) as MLTensor;
    return this;
  }

  predict(X: MLTensor): MLTensor {
    const D = addBias(X, this.fitIntercept);
    return (matmul(D, this.weight_!) as MLTensor).reshape([X.shape[0]]);
  }

  score(X: MLTensor, y: MLTensor): number {
    return r2_score(y, this.predict(X));
  }
}

export class Ridge {
  alpha: number;
  fitIntercept: boolean;
  coef_: MLTensor | null;
  intercept_: MLTensor | null;

  constructor({ alpha = 1, fitIntercept = true }: { alpha?: number; fitIntercept?: boolean } = {}) {
    this.alpha = alpha;
    this.fitIntercept = fitIntercept;
    this.coef_ = null;
    this.intercept_ = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    const d = X.shape[1];
    const y2 = asColumn(y);
    let Xc: MLTensor = X;
    let yc: MLTensor = y2;
    let meanX: MLTensor | null = null;
    let meanY: MLTensor | null = null;
    if (this.fitIntercept) {
      meanX = mean(X, [0], true) as MLTensor;
      meanY = mean(y2, [0], true) as MLTensor;
      Xc = sub(X, meanX) as MLTensor;
      yc = sub(y2, meanY) as MLTensor;
    }
    const Xt = Xc.transpose(0, 1);
    const A = add(matmul(Xt, Xc), mul(eye(d, d, { device: X.device, dtype: X.dtype }), this.alpha)) as MLTensor;
    this.coef_ = solve(A, matmul(Xt, yc)) as MLTensor;
    this.intercept_ = this.fitIntercept
      ? sub(meanY!, matmul(meanX!, this.coef_)) as MLTensor
      : zeros([1, 1], { device: X.device, dtype: X.dtype }) as MLTensor;
    return this;
  }

  predict(X: MLTensor): MLTensor {
    return (add(matmul(X, this.coef_!), this.intercept_!) as MLTensor).reshape([X.shape[0]]);
  }

  score(X: MLTensor, y: MLTensor): number {
    return r2_score(y, this.predict(X));
  }
}

export class ElasticNet {
  alpha: number;
  l1Ratio: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  coef_: MLTensor | null;
  intercept_: MLTensor | null;

  constructor({ alpha = 1, l1Ratio = 0.5, fitIntercept = true, maxIter = 1000, tol = 1e-6 }: { alpha?: number; l1Ratio?: number; fitIntercept?: boolean; maxIter?: number; tol?: number } = {}) {
    this.alpha = alpha;
    this.l1Ratio = l1Ratio;
    this.fitIntercept = fitIntercept;
    this.maxIter = maxIter;
    this.tol = tol;
    this.coef_ = null;
    this.intercept_ = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    const [coef, intercept] = _dispatch(
      'elastic_net', X, asColumn(y).reshape([X.shape[0]]),
      this.alpha, this.l1Ratio, this.maxIter, this.tol, this.fitIntercept,
    ) as [MLTensor, MLTensor];
    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: MLTensor): MLTensor {
    const w = this.coef_!.reshape([X.shape[1], 1]);
    return add((matmul(X, w) as MLTensor).reshape([X.shape[0]]), this.intercept_!) as MLTensor;
  }

  score(X: MLTensor, y: MLTensor): number {
    return r2_score(y, this.predict(X));
  }
}

export class Lasso extends ElasticNet {
  constructor({ alpha = 1, fitIntercept = true, maxIter = 1000, tol = 1e-6 }: { alpha?: number; fitIntercept?: boolean; maxIter?: number; tol?: number } = {}) {
    super({ alpha, l1Ratio: 1, fitIntercept, maxIter, tol });
  }
}

export class LogisticRegression {
  C: number;
  lr: number;
  maxIter: number;
  W_: MLTensor | null;
  b_: MLTensor | null;
  classes_: number[] | null;

  constructor({ C = 1, lr = 0.5, maxIter = 1000 }: { C?: number; lr?: number; maxIter?: number } = {}) {
    this.C = C;
    this.lr = lr;
    this.maxIter = maxIter;
    this.W_ = null;
    this.b_ = null;
    this.classes_ = null;
  }

  fit(X: MLTensor, y: MLTensor): this {
    const n = X.shape[0];
    const d = X.shape[1];
    const yv = vectorOf(y);
    const { y: labels, classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    const K = classes.length;
    const oneHot = new Float64Array(n * K);
    for (let i = 0; i < n; i++) oneHot[i * K + labels[i]] = 1;
    const Y = tensor(oneHot, { shape: [n, K], dtype: X.dtype, device: X.device }) as MLTensor;

    let W = zeros([d, K], { device: X.device, dtype: X.dtype }) as MLTensor;
    let b = zeros([1, K], { device: X.device, dtype: X.dtype }) as MLTensor;
    const reg = 1 / this.C;
    const Xt = X.transpose(0, 1);
    const step = this.lr / n;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const logits = add(matmul(X, W), b) as MLTensor;
      const shifted = sub(logits, max(logits, 1, true)) as MLTensor;
      const ex = exp(shifted) as MLTensor;
      const P = div(ex, sum(ex, [1], true)) as MLTensor;
      const G = sub(P, Y) as MLTensor;
      const gradW = add(matmul(Xt, G), mul(W, reg)) as MLTensor;
      const gradb = sum(G, [0], true) as MLTensor;
      W = sub(W, mul(gradW, step)) as MLTensor;
      b = sub(b, mul(gradb, step)) as MLTensor;
    }
    this.W_ = W;
    this.b_ = b;
    return this;
  }

  decisionLogits(X: MLTensor): MLTensor {
    return add(matmul(X, this.W_!), this.b_!) as MLTensor;
  }

  predict(X: MLTensor): MLTensor {
    const idx = argmax(this.decisionLogits(X), 1, false).toArray() as ArrayLike<number>;
    const out = new Float64Array(idx.length);
    const classes = this.classes_!;
    for (let i = 0; i < idx.length; i++) out[i] = classes[idx[i]];
    return tensor(out, { shape: [idx.length], dtype: X.dtype, device: X.device }) as MLTensor;
  }

  score(X: MLTensor, y: MLTensor): number {
    return accuracy_score(y, this.predict(X));
  }
}
