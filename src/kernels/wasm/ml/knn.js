import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { cpuKnnPredict, topKSelect, knnVote } from '../../cpu/ml/knn.js';
import { simdModule, F64_BYTES } from '../simd/runtime.js';
import { distRowWat } from '../simd/kernels.js';

const KNN_MIN_WORK = 20000;

export function wasmKnnPredict(_keySet, XTrain, yTrain, XQuery, nNeighbors, classify) {
  const tr = hostMatrix(XTrain);
  const q = hostMatrix(XQuery);
  const y = hostColumns(yTrain);
  const d = tr.cols;
  if (q.rows * tr.rows * d < KNN_MIN_WORK) {
    return cpuKnnPredict(_keySet, XTrain, yTrain, XQuery, nNeighbors, classify);
  }

  const mod = simdModule('knn_dist', distRowWat, 'dist_row');
  mod.reset();
  const qp = mod.allocF64(q.rows * d);
  const tp = mod.allocF64(tr.rows * d);
  const op = mod.allocF64(tr.rows);
  mod.writeF64(qp, q.data);
  mod.writeF64(tp, tr.data);

  const k = Math.min(nNeighbors, tr.rows);
  const out = new Float64Array(q.rows);
  const idx = new Int32Array(tr.rows);
  const rowBytes = d * F64_BYTES;
  for (let i = 0; i < q.rows; i++) {
    mod.run(qp + i * rowBytes, tp, tr.rows, d, op);
    const dist = mod.f64(op, tr.rows);
    topKSelect(dist, idx, tr.rows, k);
    out[i] = knnVote(y.data, idx, k, classify);
  }
  return toHostTensor(out, [q.rows], XQuery.dtype, XQuery.device);
}
