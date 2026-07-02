import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';

export function topKSelect(dist, idx, ntr, k) {
  for (let t = 0; t < ntr; t++) idx[t] = t;
  for (let a = 0; a < k; a++) {
    let m = a;
    for (let b = a + 1; b < ntr; b++) if (dist[idx[b]] < dist[idx[m]]) m = b;
    const tmp = idx[a]; idx[a] = idx[m]; idx[m] = tmp;
  }
}

export function knnVote(y, idx, k, classify) {
  if (classify) {
    const votes = new Map();
    let bestLabel = y[idx[0]];
    let bestCount = 0;
    for (let a = 0; a < k; a++) {
      const label = y[idx[a]];
      const c = (votes.get(label) || 0) + 1;
      votes.set(label, c);
      if (c > bestCount) { bestCount = c; bestLabel = label; }
    }
    return bestLabel;
  }
  let s = 0;
  for (let a = 0; a < k; a++) s += y[idx[a]];
  return s / k;
}

export function cpuKnnPredict(_keySet, XTrain, yTrain, XQuery, nNeighbors, classify) {
  const tr = hostMatrix(XTrain);
  const q = hostMatrix(XQuery);
  const y = hostColumns(yTrain);
  const d = tr.cols;
  const k = Math.min(nNeighbors, tr.rows);
  const out = new Float64Array(q.rows);
  const dist = new Float64Array(tr.rows);
  const idx = new Int32Array(tr.rows);

  for (let i = 0; i < q.rows; i++) {
    for (let t = 0; t < tr.rows; t++) {
      let s = 0;
      for (let j = 0; j < d; j++) {
        const diff = q.data[i * d + j] - tr.data[t * d + j];
        s += diff * diff;
      }
      dist[t] = s;
    }
    topKSelect(dist, idx, tr.rows, k);
    out[i] = knnVote(y.data, idx, k, classify);
  }
  return toHostTensor(out, [q.rows], XQuery.dtype, XQuery.device);
}
