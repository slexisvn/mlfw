import { describe, it, expect } from 'vitest';
import { tensor, ml } from '../../src/index.js';
import { WASM_DEVICE, CPU_DEVICE } from '../../src/tensor/types/device.js';
import { cpuKmeans, cpuKmeansPredict } from '../../src/backend/cpu/ml/kmeans.js';
import { wasmKmeans, wasmKmeansPredict } from '../../src/backend/wasm/ml/kmeans.js';
import { cpuKnnPredict } from '../../src/backend/cpu/ml/knn.js';
import { wasmKnnPredict } from '../../src/backend/wasm/ml/knn.js';
import { cpuElasticNet } from '../../src/backend/cpu/ml/elastic_net.js';
import { wasmElasticNet } from '../../src/backend/wasm/ml/elastic_net.js';
import { makeRng } from '../../src/ml/_random.js';

function flatWasm(data, shape) {
  return tensor(Float64Array.from(data), { shape, device: WASM_DEVICE });
}

function maxDiff(a, b) {
  const fa = a.flat(Infinity);
  const fb = b.flat(Infinity);
  let m = 0;
  for (let i = 0; i < fa.length; i++) m = Math.max(m, Math.abs(fa[i] - fb[i]));
  return m;
}

function separableBlobs(n, d, k, seed) {
  const rng = makeRng(seed);
  const data = new Float64Array(n * d);
  const labels = [];
  for (let i = 0; i < n; i++) {
    const c = i % k;
    labels.push(c);
    for (let j = 0; j < d; j++) data[i * d + j] = c * 20 + rng() * 2;
  }
  return { data, labels };
}

describe('wasm-simd ml kernels match the js reference on wasm-device tensors', () => {
  it('kmeans (above threshold) produces identical labels/centers/inertia', () => {
    const n = 600, d = 6, k = 4;
    const { data } = separableBlobs(n, d, k, 3);
    const X = flatWasm(data, [n, d]);
    const js = cpuKmeans([], X, k, 50, 1, 7);
    const wa = wasmKmeans([], X, k, 50, 1, 7);
    expect(wa[1].toArray()).toEqual(js[1].toArray());
    expect(maxDiff(wa[0].toArray(), js[0].toArray())).toBeLessThan(1e-4);
    expect(Math.abs(wa[2].toArray()[0] - js[2].toArray()[0])).toBeLessThan(1e-3);
    expect(wa[0].device.type).toBe('wasm');
  });

  it('kmeans_predict (above threshold) matches the js reference', () => {
    const n = 500, d = 8, k = 6;
    const { data } = separableBlobs(n, d, k, 5);
    const X = flatWasm(data, [n, d]);
    const centers = flatWasm(separableBlobs(k, d, k, 5).data, [k, d]);
    const js = cpuKmeansPredict([], X, centers);
    const wa = wasmKmeansPredict([], X, centers);
    expect(wa.toArray()).toEqual(js.toArray());
  });

  it('knn_predict classify (above threshold) matches the js reference', () => {
    const ntr = 400, nq = 120, d = 6;
    const trBlob = separableBlobs(ntr, d, 3, 11);
    const Xtr = flatWasm(trBlob.data, [ntr, d]);
    const ytr = flatWasm(trBlob.labels, [ntr]);
    const Xq = flatWasm(separableBlobs(nq, d, 3, 12).data, [nq, d]);
    const js = cpuKnnPredict([], Xtr, ytr, Xq, 5, true);
    const wa = wasmKnnPredict([], Xtr, ytr, Xq, 5, true);
    expect(wa.toArray()).toEqual(js.toArray());
  });

  it('knn_predict regress (above threshold) matches the js reference', () => {
    const ntr = 400, nq = 120, d = 6;
    const rng = makeRng(21);
    const Xtr = flatWasm(Array.from({ length: ntr * d }, () => rng() * 5), [ntr, d]);
    const ytr = flatWasm(Array.from({ length: ntr }, () => rng() * 3), [ntr]);
    const Xq = flatWasm(Array.from({ length: nq * d }, () => rng() * 5), [nq, d]);
    const js = cpuKnnPredict([], Xtr, ytr, Xq, 7, false);
    const wa = wasmKnnPredict([], Xtr, ytr, Xq, 7, false);
    expect(maxDiff(wa.toArray(), js.toArray())).toBeLessThan(1e-5);
  });

  it('elastic_net (above threshold) matches the js reference', () => {
    const n = 250, d = 24, maxIter = 1000;
    const rng = makeRng(31);
    const Xdata = Array.from({ length: n * d }, () => rng() * 2 - 1);
    const ydata = Array.from({ length: n }, () => rng() * 2 - 1);
    const X = flatWasm(Xdata, [n, d]);
    const y = flatWasm(ydata, [n]);
    const js = cpuElasticNet([], X, y, 0.05, 0.5, maxIter, 1e-6, true);
    const wa = wasmElasticNet([], X, y, 0.05, 0.5, maxIter, 1e-6, true);
    expect(maxDiff(wa[0].toArray(), js[0].toArray())).toBeLessThan(1e-5);
    expect(Math.abs(wa[1].toArray()[0] - js[1].toArray()[0])).toBeLessThan(1e-5);
  });

  it('small inputs fall back to the js kernel (equivalent results)', () => {
    const X = flatWasm([0, 0, 0.1, 0.1, 8, 8, 8.1, 7.9], [4, 2]);
    const js = cpuKmeans([], X, 2, 20, 1, 1);
    const wa = wasmKmeans([], X, 2, 20, 1, 1);
    expect(wa[1].toArray()).toEqual(js[1].toArray());
  });
});

describe('wasm-simd ml kernels dispatch through the public estimators', () => {
  it('KMeans on a wasm-device tensor matches a cpu-device tensor', () => {
    const n = 600, d = 6, k = 4;
    const { data } = separableBlobs(n, d, k, 9);
    const Xw = flatWasm(data, [n, d]);
    const Xc = tensor(Float64Array.from(data), { shape: [n, d], device: CPU_DEVICE });
    const kmW = new ml.KMeans({ nClusters: k, seed: 2 }).fit(Xw);
    const kmC = new ml.KMeans({ nClusters: k, seed: 2 }).fit(Xc);
    expect(kmW.labels_.toArray()).toEqual(kmC.labels_.toArray());
    expect(kmW.clusterCenters_.device.type).toBe('wasm');
  });

  it('KNeighborsClassifier on a wasm-device tensor matches a cpu-device tensor', () => {
    const ntr = 400, nq = 120, d = 6;
    const trBlob = separableBlobs(ntr, d, 3, 15);
    const qBlob = separableBlobs(nq, d, 3, 16);
    const knnW = new ml.KNeighborsClassifier({ nNeighbors: 5 })
      .fit(flatWasm(trBlob.data, [ntr, d]), flatWasm(trBlob.labels, [ntr]));
    const knnC = new ml.KNeighborsClassifier({ nNeighbors: 5 })
      .fit(tensor(Float64Array.from(trBlob.data), { shape: [ntr, d], device: CPU_DEVICE }),
        tensor(Float64Array.from(trBlob.labels), { shape: [ntr], device: CPU_DEVICE }));
    const predW = knnW.predict(flatWasm(qBlob.data, [nq, d])).toArray();
    const predC = knnC.predict(tensor(Float64Array.from(qBlob.data), { shape: [nq, d], device: CPU_DEVICE })).toArray();
    expect(predW).toEqual(predC);
  });
});
