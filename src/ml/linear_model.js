import { solve, lstsq } from '../tensor/ops/linalg.js';
import { matmul, add, sub, mul, div, exp, sum, mean, max, argmax, cat, _dispatch } from '../tensor/ops/ops.js';
import { ones, eye, zeros } from '../tensor/factory/creation_ops.js';
import { tensor } from '../tensor/factory/from_ops.js';
import { vectorOf, encodeLabels } from './_util.js';
import { r2_score, accuracy_score } from './metrics.js';

function addBias(X, fitIntercept) {
  if (!fitIntercept) return X;
  return cat([X, ones([X.shape[0], 1], { device: X.device, dtype: X.dtype })], 1);
}

function asColumn(y) {
  return y.ndim === 1 ? y.reshape([y.shape[0], 1]) : y;
}

export class LinearRegression {
  constructor({ fitIntercept = true } = {}) {
    this.fitIntercept = fitIntercept;
    this.weight_ = null;
  }

  fit(X, y) {
    const D = addBias(X, this.fitIntercept);
    this.weight_ = lstsq(D, asColumn(y));
    return this;
  }

  predict(X) {
    const D = addBias(X, this.fitIntercept);
    return matmul(D, this.weight_).reshape([X.shape[0]]);
  }

  score(X, y) {
    return r2_score(y, this.predict(X));
  }
}

export class Ridge {
  constructor({ alpha = 1, fitIntercept = true } = {}) {
    this.alpha = alpha;
    this.fitIntercept = fitIntercept;
    this.coef_ = null;
    this.intercept_ = null;
  }

  fit(X, y) {
    const d = X.shape[1];
    const y2 = asColumn(y);
    let Xc = X;
    let yc = y2;
    let meanX = null;
    let meanY = null;
    if (this.fitIntercept) {
      meanX = mean(X, [0], true);
      meanY = mean(y2, [0], true);
      Xc = sub(X, meanX);
      yc = sub(y2, meanY);
    }
    const Xt = Xc.transpose(0, 1);
    const A = add(matmul(Xt, Xc), mul(eye(d, d, { device: X.device, dtype: X.dtype }), this.alpha));
    this.coef_ = solve(A, matmul(Xt, yc));
    this.intercept_ = this.fitIntercept
      ? sub(meanY, matmul(meanX, this.coef_))
      : zeros([1, 1], { device: X.device, dtype: X.dtype });
    return this;
  }

  predict(X) {
    return add(matmul(X, this.coef_), this.intercept_).reshape([X.shape[0]]);
  }

  score(X, y) {
    return r2_score(y, this.predict(X));
  }
}

export class ElasticNet {
  constructor({ alpha = 1, l1Ratio = 0.5, fitIntercept = true, maxIter = 1000, tol = 1e-6 } = {}) {
    this.alpha = alpha;
    this.l1Ratio = l1Ratio;
    this.fitIntercept = fitIntercept;
    this.maxIter = maxIter;
    this.tol = tol;
    this.coef_ = null;
    this.intercept_ = null;
  }

  fit(X, y) {
    const [coef, intercept] = _dispatch(
      'elastic_net', X, asColumn(y).reshape([X.shape[0]]),
      this.alpha, this.l1Ratio, this.maxIter, this.tol, this.fitIntercept,
    );
    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X) {
    const w = this.coef_.reshape([X.shape[1], 1]);
    return add(matmul(X, w).reshape([X.shape[0]]), this.intercept_);
  }

  score(X, y) {
    return r2_score(y, this.predict(X));
  }
}

export class Lasso extends ElasticNet {
  constructor({ alpha = 1, fitIntercept = true, maxIter = 1000, tol = 1e-6 } = {}) {
    super({ alpha, l1Ratio: 1, fitIntercept, maxIter, tol });
  }
}

export class LogisticRegression {
  constructor({ C = 1, lr = 0.5, maxIter = 1000 } = {}) {
    this.C = C;
    this.lr = lr;
    this.maxIter = maxIter;
    this.W_ = null;
    this.b_ = null;
    this.classes_ = null;
  }

  fit(X, y) {
    const n = X.shape[0];
    const d = X.shape[1];
    const yv = vectorOf(y);
    const { y: labels, classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    const K = classes.length;
    const oneHot = new Float64Array(n * K);
    for (let i = 0; i < n; i++) oneHot[i * K + labels[i]] = 1;
    const Y = tensor(oneHot, { shape: [n, K], dtype: X.dtype, device: X.device });

    let W = zeros([d, K], { device: X.device, dtype: X.dtype });
    let b = zeros([1, K], { device: X.device, dtype: X.dtype });
    const reg = 1 / this.C;
    const Xt = X.transpose(0, 1);
    const step = this.lr / n;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const logits = add(matmul(X, W), b);
      const shifted = sub(logits, max(logits, 1, true));
      const ex = exp(shifted);
      const P = div(ex, sum(ex, [1], true));
      const G = sub(P, Y);
      const gradW = add(matmul(Xt, G), mul(W, reg));
      const gradb = sum(G, [0], true);
      W = sub(W, mul(gradW, step));
      b = sub(b, mul(gradb, step));
    }
    this.W_ = W;
    this.b_ = b;
    return this;
  }

  decisionLogits(X) {
    return add(matmul(X, this.W_), this.b_);
  }

  predict(X) {
    const idx = argmax(this.decisionLogits(X), 1, false).toArray();
    const out = new Float64Array(idx.length);
    for (let i = 0; i < idx.length; i++) out[i] = this.classes_[idx[i]];
    return tensor(out, { shape: [idx.length], dtype: X.dtype, device: X.device });
  }

  score(X, y) {
    return accuracy_score(y, this.predict(X));
  }
}
