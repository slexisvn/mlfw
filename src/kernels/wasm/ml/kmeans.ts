import { hostMatrix, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { makeRng } from '../../../ml/_random.js';
import { cpuKmeans, cpuKmeansPredict, kmeansPlusPlus, updateCenters } from '../../cpu/ml/kmeans.js';
import { simdModule } from '../simd/runtime.js';
import { nearestCentroidWat } from '../simd/kernels.js';
import type { SimdModule } from '../simd/runtime.js';
import type { KmeansFit } from '../../cpu/ml/kmeans.js';
import type { Tensor } from '../../../tensor/core/tensor.js';

const KMEANS_MIN_WORK = 8000;
const PREDICT_MIN_WORK = 50000;
const SEED_STRIDE = 0x9e3779b9;

function assignWasm(mod: SimdModule, xp: number, cp: number, lp: number, ip: number, n: number, d: number, k: number, centers: Float64Array, labels: Int32Array): number {
  mod.writeF64(cp, centers);
  mod.run(xp, n, d, cp, k, lp, ip);
  labels.set(mod.i32(lp, n));
  return mod.f64(ip, 1)[0];
}

export function wasmKmeans(_keySet: unknown, X: Tensor, nClusters: number, maxIter: number, nInit: number, seed: number): Tensor[] {
  const { data, rows: n, cols: d } = hostMatrix(X);
  if (n * d * nClusters < KMEANS_MIN_WORK) return cpuKmeans(_keySet, X, nClusters, maxIter, nInit, seed);

  const mod = simdModule('kmeans_assign', nearestCentroidWat, 'nearest_centroid');
  let best: KmeansFit | null = null;
  for (let run = 0; run < nInit; run++) {
    mod.reset();
    const xp = mod.allocF64(n * d);
    const cp = mod.allocF64(nClusters * d);
    const lp = mod.allocI32(n);
    const ip = mod.allocF64(1);
    mod.writeF64(xp, data);

    const rng = makeRng(seed + run * SEED_STRIDE);
    const centers = kmeansPlusPlus(data, n, d, nClusters, rng);
    const labels = new Int32Array(n);
    let inertia = Infinity;
    for (let iter = 0; iter < maxIter; iter++) {
      inertia = assignWasm(mod, xp, cp, lp, ip, n, d, nClusters, centers, labels);
      if (!updateCenters(data, n, d, nClusters, labels, centers)) break;
    }
    inertia = assignWasm(mod, xp, cp, lp, ip, n, d, nClusters, centers, labels);
    if (!best || inertia < best.inertia) {
      best = { centers: Float64Array.from(centers), labels: Int32Array.from(labels), inertia };
    }
  }

  const labelsOut = new Float64Array(n);
  for (let i = 0; i < n; i++) labelsOut[i] = (best as KmeansFit).labels[i];
  return [
    toHostTensor((best as KmeansFit).centers, [nClusters, d], X.dtype, X.device),
    toHostTensor(labelsOut, [n], X.dtype, X.device),
    toHostTensor(new Float64Array([(best as KmeansFit).inertia]), [1], X.dtype, X.device),
  ];
}

export function wasmKmeansPredict(_keySet: unknown, X: Tensor, centersT: Tensor): Tensor {
  const { data, rows: n, cols: d } = hostMatrix(X);
  const c = hostMatrix(centersT);
  if (n * d * c.rows < PREDICT_MIN_WORK) return cpuKmeansPredict(_keySet, X, centersT);

  const mod = simdModule('kmeans_assign', nearestCentroidWat, 'nearest_centroid');
  mod.reset();
  const xp = mod.allocF64(n * d);
  const cp = mod.allocF64(c.rows * d);
  const lp = mod.allocI32(n);
  const ip = mod.allocF64(1);
  mod.writeF64(xp, data);
  mod.writeF64(cp, c.data);
  mod.run(xp, n, d, cp, c.rows, lp, ip);
  const labels = new Float64Array(n);
  const li = mod.i32(lp, n);
  for (let i = 0; i < n; i++) labels[i] = li[i];
  return toHostTensor(labels, [n], X.dtype, X.device);
}
