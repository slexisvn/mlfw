import * as ops from '../tensor/ops/ops.js';

function _gradParams(parameters) {
  const params = Array.isArray(parameters) ? parameters : [...parameters];
  return params.filter(p => p.grad !== null);
}

export function clipGradNorm_(parameters, maxNorm, normType = 2) {
  const params = _gradParams(parameters);
  if (params.length === 0) return 0;

  let totalNorm;
  if (normType === 2) {
    let acc = null;
    for (const p of params) {
      const sq = ops.sum(ops.mul(p.grad, p.grad));
      acc = acc === null ? sq : ops.add(acc, sq);
    }
    totalNorm = Math.sqrt(acc.item());
  } else if (normType === Infinity) {
    let acc = null;
    for (const p of params) {
      const m = ops.max(ops.abs(p.grad));
      acc = acc === null ? m : ops.maximum(acc, m);
    }
    totalNorm = acc.item();
  } else {
    let acc = null;
    for (const p of params) {
      const s = ops.sum(ops.pow(ops.abs(p.grad), normType));
      acc = acc === null ? s : ops.add(acc, s);
    }
    totalNorm = Math.pow(acc.item(), 1 / normType);
  }

  const clipCoef = maxNorm / (totalNorm + 1e-6);
  if (clipCoef < 1) {
    for (const p of params) p.grad = ops.mul(p.grad, clipCoef);
  }
  return totalNorm;
}

export function clipGradValue_(parameters, clipValue) {
  for (const p of _gradParams(parameters)) {
    p.grad = ops.clamp(p.grad, -clipValue, clipValue);
  }
}
