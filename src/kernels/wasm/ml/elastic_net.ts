import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { cpuElasticNet, elasticNetPrep, elasticNetIntercept } from '../../cpu/ml/elastic_net.js';
import { simdModule } from '../simd/runtime.js';
import { coordDescentWat } from '../simd/kernels.js';
import type { Tensor } from '../../../tensor/core/tensor.js';

const ELASTIC_NET_MIN_WORK = 3000000;

export function wasmElasticNet(_keySet: unknown, X: Tensor, y: Tensor, alpha: number, l1Ratio: number, maxIter: number, tol: number, fitIntercept: boolean): Tensor[] {
  const { data, rows: n, cols: d } = hostMatrix(X);
  if (n * d * maxIter < ELASTIC_NET_MIN_WORK) {
    return cpuElasticNet(_keySet, X, y, alpha, l1Ratio, maxIter, tol, fitIntercept);
  }
  const yv = hostColumns(y);
  const { xc, r, z, meanX, meanY } = elasticNetPrep(data, n, d, yv.data, fitIntercept);
  const xcCol = new Float64Array(n * d);
  for (let j = 0; j < d; j++) {
    for (let i = 0; i < n; i++) xcCol[j * n + i] = xc[i * d + j];
  }
  const l1 = alpha * l1Ratio;
  const l2 = alpha * (1 - l1Ratio);

  const mod = simdModule('coord_descent', coordDescentWat, 'coord_descent');
  mod.reset();
  const xcp = mod.allocF64(n * d);
  const rp = mod.allocF64(n);
  const wp = mod.allocF64(d);
  const zp = mod.allocF64(d);
  mod.writeF64(xcp, xcCol);
  mod.writeF64(rp, r);
  mod.writeF64(zp, z);
  mod.f64(wp, d).fill(0);
  mod.run(xcp, n, d, rp, wp, zp, l1, l2, maxIter, tol);
  const w = Float64Array.from(mod.f64(wp, d));

  const intercept = elasticNetIntercept(meanX, meanY, w, d);
  return [
    toHostTensor(w, [d], X.dtype, X.device),
    toHostTensor(new Float64Array([intercept]), [1], X.dtype, X.device),
  ];
}
