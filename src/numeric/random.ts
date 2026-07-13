import { makeRng } from '../ml/_random.js';
import { normalPpfScalar } from './special.js';
import { cholesky } from '../tensor/ops/linalg.js';
import { matmul, add } from '../tensor/ops/ops.js';
import { toHostTensor } from '../tensor/utils/host_matrix.js';
import { hostVector, hostGrid } from './_array.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { NumericMatrixInput, NumericShape, NumericVectorInput, TensorOptions } from './types.js';

const MT_SQUEEZE = 0.0331;

type RandomOptions = TensorOptions & { low?: number; high?: number; loc?: number; scale?: number; df?: number };
type TransposableTensor = Tensor & { transpose(dim0: number, dim1: number): Tensor };

function normalizeShape(shape: NumericShape): { shp: number[]; n: number } {
  const shp = Array.isArray(shape) ? shape : [shape];
  let n = 1;
  for (const d of shp) n *= d;
  return { shp, n };
}

export class Generator {
  private readonly _next: () => number;

  constructor(seed?: number) {
    this._next = makeRng(seed);
  }

  _uniformPositive(): number {
    let u = this._next();
    while (u <= 0) u = this._next();
    return u;
  }

  _normalDraw(): number {
    return normalPpfScalar(this._uniformPositive());
  }

  _gammaDraw(a: number): number {
    if (a < 1) return this._gammaDraw(a + 1) * Math.pow(this._uniformPositive(), 1 / a);
    const d = a - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let z;
      let v;
      do {
        z = this._normalDraw();
        v = 1 + c * z;
      } while (v <= 0);
      v = v * v * v;
      const u = this._uniformPositive();
      if (u < 1 - MT_SQUEEZE * z * z * z * z) return d * v;
      if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  _fill(shape: NumericShape, opts: TensorOptions, draw: () => number): Tensor {
    const { shp, n } = normalizeShape(shape);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = draw();
    return toHostTensor(out, shp, opts.dtype ?? 'f64', opts.device);
  }

  uniform(shape: NumericShape, opts: RandomOptions = {}): Tensor {
    const low = opts.low ?? 0;
    const high = opts.high ?? 1;
    return this._fill(shape, opts, () => low + (high - low) * this._next());
  }

  normal(shape: NumericShape, opts: RandomOptions = {}): Tensor {
    const loc = opts.loc ?? 0;
    const scale = opts.scale ?? 1;
    return this._fill(shape, opts, () => loc + scale * this._normalDraw());
  }

  standardT(shape: NumericShape, opts: RandomOptions = {}): Tensor {
    const df = opts.df as number;
    return this._fill(shape, opts, () => {
      const z = this._normalDraw();
      const g = 2 * this._gammaDraw(df / 2);
      return z / Math.sqrt(g / df);
    });
  }

  chi2(shape: NumericShape, opts: RandomOptions = {}): Tensor {
    const df = opts.df as number;
    return this._fill(shape, opts, () => 2 * this._gammaDraw(df / 2));
  }

  exponential(shape: NumericShape, opts: RandomOptions = {}): Tensor {
    const scale = opts.scale ?? 1;
    return this._fill(shape, opts, () => -scale * Math.log(this._uniformPositive()));
  }

  multivariateNormal(mean: NumericVectorInput, cov: NumericMatrixInput, n = 1, opts: TensorOptions = {}): Tensor {
    const meanV = hostVector(mean);
    const covG = hostGrid(cov);
    if (covG.rows !== covG.cols || covG.rows !== meanV.data.length) {
      throw new Error(`multivariateNormal: mean of length ${meanV.data.length} incompatible with ${covG.rows}x${covG.cols} covariance`);
    }
    const d = covG.rows;
    const device = opts.device ?? covG.device;
    const dtype = opts.dtype ?? 'f64';
    const L = cholesky(toHostTensor(covG.data, [d, d], dtype, device));
    const z = this.normal([n, d], { device, dtype });
    const meanT = toHostTensor(meanV.data, [d], dtype, device);
    return add(matmul(z, (L as TransposableTensor).transpose(0, 1)), meanT) as Tensor;
  }
}
