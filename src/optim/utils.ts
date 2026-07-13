import * as ops from '../tensor/ops/ops.js';
import type { Tensor } from '../tensor/core/tensor.js';

function _gradParams(parameters: Iterable<Tensor> | Tensor[]): Tensor[] {
  const params = Array.isArray(parameters) ? parameters : [...parameters];
  return params.filter(p => p.grad !== null);
}

export function clipGradNorm_(parameters: Iterable<Tensor> | Tensor[], maxNorm: number, normType = 2): number {
  const params = _gradParams(parameters);
  if (params.length === 0) return 0;

  let totalNorm: number;
  if (normType === 2) {
    let acc: Tensor | null = null;
    for (const p of params) {
      const sq = ops.sum(ops.mul(p.grad!, p.grad!));
      acc = acc === null ? sq : ops.add(acc, sq);
    }
    totalNorm = Math.sqrt(Number(acc!.item()));
  } else if (normType === Infinity) {
    let acc: Tensor | null = null;
    for (const p of params) {
      const m = ops.max(ops.abs(p.grad!));
      acc = acc === null ? m : ops.maximum(acc, m);
    }
    totalNorm = Number(acc!.item());
  } else {
    let acc: Tensor | null = null;
    for (const p of params) {
      const s = ops.sum(ops.pow(ops.abs(p.grad!), normType));
      acc = acc === null ? s : ops.add(acc, s);
    }
    totalNorm = Math.pow(Number(acc!.item()), 1 / normType);
  }

  const clipCoef = maxNorm / (totalNorm + 1e-6);
  if (clipCoef < 1) {
    for (const p of params) p.grad = ops.mul(p.grad!, clipCoef);
  }
  return totalNorm;
}

export function clipGradValue_(parameters: Iterable<Tensor> | Tensor[], clipValue: number): void {
  for (const p of _gradParams(parameters)) {
    p.grad = ops.clamp(p.grad!, -clipValue, clipValue);
  }
}
