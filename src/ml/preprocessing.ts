import { matrixOf, matrix, vectorOf, vector, encodeLabels } from './_util.js';
import type { MLTensor } from './types.js';

export class StandardScaler {
  withMean: boolean;
  withStd: boolean;
  mean_: Float64Array | null;
  scale_: Float64Array | null;
  private _cols?: number;

  constructor({ withMean = true, withStd = true }: { withMean?: boolean; withStd?: boolean } = {}) {
    this.withMean = withMean;
    this.withStd = withStd;
    this.mean_ = null;
    this.scale_ = null;
  }

  fit(X: MLTensor): this {
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

  transform(X: MLTensor): MLTensor {
    const m = matrixOf(X);
    const mean = this.mean_!;
    const scale = this.scale_!;
    const out = new Float64Array(m.rows * m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        out[i * m.cols + j] = (m.data[i * m.cols + j] - mean[j]) / scale[j];
      }
    }
    return matrix(out, m.rows, m.cols, X.dtype);
  }

  fit_transform(X: MLTensor): MLTensor {
    return this.fit(X).transform(X);
  }

  inverse_transform(X: MLTensor): MLTensor {
    const m = matrixOf(X);
    const mean = this.mean_!;
    const scale = this.scale_!;
    const out = new Float64Array(m.rows * m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        out[i * m.cols + j] = m.data[i * m.cols + j] * scale[j] + mean[j];
      }
    }
    return matrix(out, m.rows, m.cols, X.dtype);
  }
}

export class LabelEncoder {
  classes_: number[] | null;
  private _lookup: Map<number, number> | null;

  constructor() {
    this.classes_ = null;
    this._lookup = null;
  }

  fit(y: MLTensor): this {
    const yv = vectorOf(y);
    const { classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    this._lookup = new Map(classes.map((c, i) => [c, i]));
    return this;
  }

  transform(y: MLTensor): MLTensor {
    const yv = vectorOf(y);
    const lookup = this._lookup!;
    const out = new Float64Array(yv.n);
    for (let i = 0; i < yv.n; i++) {
      const idx = lookup.get(yv.data[i]);
      if (idx === undefined) throw new Error(`LabelEncoder: unseen label ${yv.data[i]}`);
      out[i] = idx;
    }
    return vector(out, yv.n, y.dtype);
  }

  fit_transform(y: MLTensor): MLTensor {
    const yv = vectorOf(y);
    const { y: encoded, classes } = encodeLabels(yv.data, yv.n);
    this.classes_ = classes;
    this._lookup = new Map(classes.map((c, i) => [c, i]));
    const out = new Float64Array(yv.n);
    for (let i = 0; i < yv.n; i++) out[i] = encoded[i];
    return vector(out, yv.n, y.dtype);
  }

  inverse_transform(y: MLTensor): number[] {
    const yv = vectorOf(y);
    const classes = this.classes_!;
    return Array.from({ length: yv.n }, (_, i) => classes[Math.round(yv.data[i])]);
  }
}

export class OneHotEncoder {
  classes_: number[] | null;

  constructor() {
    this.classes_ = null;
  }

  fit(y: MLTensor): this {
    const yv = vectorOf(y);
    this.classes_ = encodeLabels(yv.data, yv.n).classes;
    return this;
  }

  transform(y: MLTensor): MLTensor {
    const yv = vectorOf(y);
    const classes = this.classes_!;
    const lookup = new Map(classes.map((c, i) => [c, i]));
    const K = classes.length;
    const out = new Float64Array(yv.n * K);
    for (let i = 0; i < yv.n; i++) {
      const idx = lookup.get(yv.data[i]);
      if (idx === undefined) throw new Error(`OneHotEncoder: unseen label ${yv.data[i]}`);
      out[i * K + idx] = 1;
    }
    return matrix(out, yv.n, K, y.dtype);
  }

  fit_transform(y: MLTensor): MLTensor {
    return this.fit(y).transform(y);
  }
}

export class MinMaxScaler {
  featureRange: readonly [number, number];
  min_: null;
  dataMin_: Float64Array | null;
  dataRange_: Float64Array | null;

  constructor({ featureRange = [0, 1] }: { featureRange?: readonly [number, number] } = {}) {
    this.featureRange = featureRange;
    this.min_ = null;
    this.dataMin_ = null;
    this.dataRange_ = null;
  }

  fit(X: MLTensor): this {
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

  transform(X: MLTensor): MLTensor {
    const m = matrixOf(X);
    const [lo, hi] = this.featureRange;
    const dataMin = this.dataMin_!;
    const dataRange = this.dataRange_!;
    const span = hi - lo;
    const out = new Float64Array(m.rows * m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        const scaled = (m.data[i * m.cols + j] - dataMin[j]) / dataRange[j];
        out[i * m.cols + j] = scaled * span + lo;
      }
    }
    return matrix(out, m.rows, m.cols, X.dtype);
  }

  fit_transform(X: MLTensor): MLTensor {
    return this.fit(X).transform(X);
  }
}
