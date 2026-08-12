import { hostMatrix, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { makeRng, randInt } from '../../../ml/_random.js';
import type { Rng } from '../../../ml/types.js';
import type { Tensor } from '../../../tensor/core/tensor.js';

export type KmeansFit = { centers: Float64Array; labels: Int32Array; inertia: number };

export function sqDist(a: Float64Array, ai: number, b: Float64Array, bi: number, d: number): number {
  let s = 0;
  for (let j = 0; j < d; j++) {
    const diff = a[ai * d + j] - b[bi * d + j];
    s += diff * diff;
  }
  return s;
}

export function kmeansPlusPlus(x: Float64Array, n: number, d: number, k: number, rng: Rng): Float64Array {
  const centers = new Float64Array(k * d);
  const first = randInt(rng, n);
  for (let j = 0; j < d; j++) centers[j] = x[first * d + j];
  const closest = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      const dist = sqDist(x, i, centers, c - 1, d);
      if (dist < closest[i]) closest[i] = dist;
      total += closest[i];
    }
    let target = rng() * total;
    let chosen = n - 1;
    for (let i = 0; i < n; i++) {
      target -= closest[i];
      if (target <= 0) { chosen = i; break; }
    }
    for (let j = 0; j < d; j++) centers[c * d + j] = x[chosen * d + j];
  }
  return centers;
}

export function assign(x: Float64Array, n: number, d: number, centers: Float64Array, k: number, labels: Int32Array): number {
  let inertia = 0;
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < k; c++) {
      const dist = sqDist(x, i, centers, c, d);
      if (dist < bestDist) { bestDist = dist; best = c; }
    }
    labels[i] = best;
    inertia += bestDist;
  }
  return inertia;
}

export function updateCenters(x: Float64Array, n: number, d: number, k: number, labels: Int32Array, centers: Float64Array): boolean {
  const sums = new Float64Array(k * d);
  const counts = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    const c = labels[i];
    counts[c]++;
    for (let j = 0; j < d; j++) sums[c * d + j] += x[i * d + j];
  }
  let moved = false;
  for (let c = 0; c < k; c++) {
    if (counts[c] === 0) continue;
    for (let j = 0; j < d; j++) {
      const v = sums[c * d + j] / counts[c];
      if (v !== centers[c * d + j]) moved = true;
      centers[c * d + j] = v;
    }
  }
  return moved;
}

function lloyd(x: Float64Array, n: number, d: number, k: number, maxIter: number, rng: Rng): KmeansFit {
  const centers = kmeansPlusPlus(x, n, d, k, rng);
  const labels = new Int32Array(n);
  let inertia = Infinity;
  for (let iter = 0; iter < maxIter; iter++) {
    inertia = assign(x, n, d, centers, k, labels);
    if (!updateCenters(x, n, d, k, labels, centers)) break;
  }
  inertia = assign(x, n, d, centers, k, labels);
  return { centers, labels, inertia };
}

export function cpuKmeans(_keySet: unknown, X: Tensor, nClusters: number, maxIter: number, nInit: number, seed: number): Tensor[] {
  const { data, rows: n, cols: d } = hostMatrix(X);
  let best: KmeansFit | null = null;
  for (let run = 0; run < nInit; run++) {
    const rng = makeRng(seed + run * 0x9e3779b9);
    const result = lloyd(data, n, d, nClusters, maxIter, rng);
    if (!best || result.inertia < best.inertia) best = result;
  }
  const labels = new Float64Array(n);
  for (let i = 0; i < n; i++) labels[i] = (best as KmeansFit).labels[i];
  return [
    toHostTensor((best as KmeansFit).centers, [nClusters, d], X.dtype, X.device),
    toHostTensor(labels, [n], X.dtype, X.device),
    toHostTensor(new Float64Array([(best as KmeansFit).inertia]), [1], X.dtype, X.device),
  ];
}

export function cpuKmeansPredict(_keySet: unknown, X: Tensor, centersT: Tensor): Tensor {
  const { data, rows: n, cols: d } = hostMatrix(X);
  const c = hostMatrix(centersT);
  const labels = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestDist = Infinity;
    for (let cc = 0; cc < c.rows; cc++) {
      const dist = sqDist(data, i, c.data, cc, d);
      if (dist < bestDist) { bestDist = dist; best = cc; }
    }
    labels[i] = best;
  }
  return toHostTensor(labels, [n], X.dtype, X.device);
}
