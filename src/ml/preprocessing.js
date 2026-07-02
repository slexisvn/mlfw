import { matrixOf, matrix, vectorOf, vector, encodeLabels } from './_util.js';

export class StandardScaler {
  constructor({ withMean = true, withStd = true } = {}) {
    this.withMean = withMean;
    this.withStd = withStd;
    this.mean_ = null;
    this.scale_ = null;
  }

  fit(X) {
    const m = matrixOf(X);
    this.mean_ = new Float64Array(m.cols);
    this.scale_ = new Float64Array(m.cols);
    for (let j = 0; j < m.cols; j++) {
      let mean = 0;
      for (let i = 0; i < m.rows; i++) mean += m.data[i * m.cols + j];
      mean /= m.rows;
      let varr = 0;
      for (let i = 0; i < m.rows; i++) {
        const d = m.data[i * m.cols + j] - mean;
        varr += d * d;
      }
      varr /= m.rows;
      const std = Math.sqrt(varr);
      this.mean_[j] = this.withMean ? mean : 0;
      this.scale_[j] = this.withStd && std > 0 ? std : 1;
    }
    this._cols = m.cols;
    return this;
  }

  transform(X) {
    const m = matrixOf(X);
    const out = new Float64Array(m.rows * m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        out[i * m.cols + j] = (m.data[i * m.cols + j] - this.mean_[j]) / this.scale_[j];
      }
    }
    return matrix(out, m.rows, m.cols, X.dtype);
  }

  fit_transform(X) {
    return this.fit(X).transform(X);
  }

  inverse_transform(X) {
    const m = matrixOf(X);
    const out = new Float64Array(m.rows * m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        out[i * m.cols + j] = m.data[i * m.cols + j] * this.scale_[j] + this.mean_[j];
      }
    }
    return matrix(out, m.rows, m.cols, X.dtype);
  }
}

export class LabelEncoder {
  constructor() {
    this.classes_ = null;
    this._lookup = null;
  }

  fit(y) {
    const yv = vectorOf(y);
    const { classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    this._lookup = new Map(classes.map((c, i) => [c, i]));
    return this;
  }

  transform(y) {
    const yv = vectorOf(y);
    const out = new Float64Array(yv.n);
    for (let i = 0; i < yv.n; i++) {
      const idx = this._lookup.get(yv.data[i]);
      if (idx === undefined) throw new Error(`LabelEncoder: unseen label ${yv.data[i]}`);
      out[i] = idx;
    }
    return vector(out, yv.n, y.dtype);
  }

  fit_transform(y) {
    const yv = vectorOf(y);
    const { y: encoded, classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    this._lookup = new Map(classes.map((c, i) => [c, i]));
    const out = new Float64Array(yv.n);
    for (let i = 0; i < yv.n; i++) out[i] = encoded[i];
    return vector(out, yv.n, y.dtype);
  }

  inverse_transform(y) {
    const yv = vectorOf(y);
    return Array.from({ length: yv.n }, (_, i) => this.classes_[Math.round(yv.data[i])]);
  }
}

export class OneHotEncoder {
  constructor() {
    this.classes_ = null;
  }

  fit(y) {
    const yv = vectorOf(y);
    this.classes_ = encodeLabels(yv.data, yv.n).classes;
    return this;
  }

  transform(y) {
    const yv = vectorOf(y);
    const lookup = new Map(this.classes_.map((c, i) => [c, i]));
    const K = this.classes_.length;
    const out = new Float64Array(yv.n * K);
    for (let i = 0; i < yv.n; i++) {
      const idx = lookup.get(yv.data[i]);
      if (idx === undefined) throw new Error(`OneHotEncoder: unseen label ${yv.data[i]}`);
      out[i * K + idx] = 1;
    }
    return matrix(out, yv.n, K, y.dtype);
  }

  fit_transform(y) {
    return this.fit(y).transform(y);
  }
}

export class MinMaxScaler {
  constructor({ featureRange = [0, 1] } = {}) {
    this.featureRange = featureRange;
    this.min_ = null;
    this.dataMin_ = null;
    this.dataRange_ = null;
  }

  fit(X) {
    const m = matrixOf(X);
    this.dataMin_ = new Float64Array(m.cols);
    this.dataRange_ = new Float64Array(m.cols);
    for (let j = 0; j < m.cols; j++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < m.rows; i++) {
        const v = m.data[i * m.cols + j];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      this.dataMin_[j] = lo;
      this.dataRange_[j] = hi > lo ? hi - lo : 1;
    }
    return this;
  }

  transform(X) {
    const m = matrixOf(X);
    const [lo, hi] = this.featureRange;
    const span = hi - lo;
    const out = new Float64Array(m.rows * m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        const scaled = (m.data[i * m.cols + j] - this.dataMin_[j]) / this.dataRange_[j];
        out[i * m.cols + j] = scaled * span + lo;
      }
    }
    return matrix(out, m.rows, m.cols, X.dtype);
  }

  fit_transform(X) {
    return this.fit(X).transform(X);
  }
}
