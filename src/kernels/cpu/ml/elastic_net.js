import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';

export function softThreshold(x, t) {
  if (x > t) return x - t;
  if (x < -t) return x + t;
  return 0;
}

export function elasticNetPrep(data, n, d, yData, fitIntercept) {
  const meanX = new Float64Array(d);
  let meanY = 0;
  if (fitIntercept) {
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += data[i * d + j];
      meanX[j] = s / n;
    }
    for (let i = 0; i < n; i++) meanY += yData[i];
    meanY /= n;
  }
  const xc = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) xc[i * d + j] = data[i * d + j] - meanX[j];
  }
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[i] = yData[i] - meanY;
  const z = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += xc[i * d + j] * xc[i * d + j];
    z[j] = s / n;
  }
  return { xc, r, z, meanX, meanY };
}

export function elasticNetIntercept(meanX, meanY, w, d) {
  let intercept = meanY;
  for (let j = 0; j < d; j++) intercept -= meanX[j] * w[j];
  return intercept;
}

function coordinateDescent(xc, n, d, r, z, w, l1, l2, maxIter, tol) {
  for (let iter = 0; iter < maxIter; iter++) {
    let maxChange = 0;
    for (let j = 0; j < d; j++) {
      if (z[j] === 0) continue;
      let dot = 0;
      for (let i = 0; i < n; i++) dot += xc[i * d + j] * r[i];
      const rho = dot / n + w[j] * z[j];
      const wj = softThreshold(rho, l1) / (z[j] + l2);
      const delta = wj - w[j];
      if (delta !== 0) {
        for (let i = 0; i < n; i++) r[i] -= delta * xc[i * d + j];
        w[j] = wj;
        if (Math.abs(delta) > maxChange) maxChange = Math.abs(delta);
      }
    }
    if (maxChange < tol) break;
  }
}

export function cpuElasticNet(_keySet, X, y, alpha, l1Ratio, maxIter, tol, fitIntercept) {
  const { data, rows: n, cols: d } = hostMatrix(X);
  const yv = hostColumns(y);
  const { xc, r, z, meanX, meanY } = elasticNetPrep(data, n, d, yv.data, fitIntercept);
  const w = new Float64Array(d);
  const l1 = alpha * l1Ratio;
  const l2 = alpha * (1 - l1Ratio);
  coordinateDescent(xc, n, d, r, z, w, l1, l2, maxIter, tol);
  const intercept = elasticNetIntercept(meanX, meanY, w, d);
  return [toHostTensor(w, [d], X.dtype, X.device), toHostTensor(new Float64Array([intercept]), [1], X.dtype, X.device)];
}
