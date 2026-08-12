import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { encodeLabels } from '../../../ml/_util.js';
import type { Tensor } from '../../../tensor/core/tensor.js';

const VAR_SMOOTHING = 1e-9;

export function cpuGaussianNbFit(_keySet: unknown, X: Tensor, y: Tensor): Tensor[] {
  const { data, rows: n, cols: d } = hostMatrix(X);
  const yv = hostColumns(y);
  const { y: labels, classes } = encodeLabels(yv.data, yv.rows);
  const K = classes.length;
  const means = new Float64Array(K * d);
  const vars = new Float64Array(K * d);
  const priors = new Float64Array(K);
  const counts = new Int32Array(K);

  for (let i = 0; i < n; i++) {
    const c = labels[i];
    counts[c]++;
    for (let j = 0; j < d; j++) means[c * d + j] += data[i * d + j];
  }
  for (let c = 0; c < K; c++) {
    for (let j = 0; j < d; j++) means[c * d + j] /= counts[c];
    priors[c] = counts[c] / n;
  }
  for (let i = 0; i < n; i++) {
    const c = labels[i];
    for (let j = 0; j < d; j++) {
      const diff = data[i * d + j] - means[c * d + j];
      vars[c * d + j] += diff * diff;
    }
  }
  let maxVar = 0;
  for (let c = 0; c < K; c++) {
    for (let j = 0; j < d; j++) {
      vars[c * d + j] /= counts[c];
      if (vars[c * d + j] > maxVar) maxVar = vars[c * d + j];
    }
  }
  const eps = VAR_SMOOTHING * maxVar;
  for (let i = 0; i < K * d; i++) vars[i] += eps;

  const classesArr = new Float64Array(K);
  for (let c = 0; c < K; c++) classesArr[c] = classes[c];
  return [
    toHostTensor(means, [K, d], X.dtype, X.device),
    toHostTensor(vars, [K, d], X.dtype, X.device),
    toHostTensor(priors, [K], X.dtype, X.device),
    toHostTensor(classesArr, [K], X.dtype, X.device),
  ];
}

export function cpuGaussianNbPredict(_keySet: unknown, X: Tensor, meansT: Tensor, varsT: Tensor, priorsT: Tensor, classesT: Tensor): Tensor {
  const { data, rows: n, cols: d } = hostMatrix(X);
  const means = hostMatrix(meansT);
  const vars = hostMatrix(varsT);
  const priors = hostColumns(priorsT);
  const classes = hostColumns(classesT);
  const K = means.rows;
  const out = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestLp = -Infinity;
    for (let c = 0; c < K; c++) {
      let lp = Math.log(priors.data[c]);
      for (let j = 0; j < d; j++) {
        const v = vars.data[c * d + j];
        const diff = data[i * d + j] - means.data[c * d + j];
        lp += -0.5 * (Math.log(2 * Math.PI * v) + (diff * diff) / v);
      }
      if (lp > bestLp) { bestLp = lp; best = c; }
    }
    out[i] = classes.data[best];
  }
  return toHostTensor(out, [n], X.dtype, X.device);
}
