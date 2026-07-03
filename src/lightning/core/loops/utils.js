export function resolveLimit(limitConfig, totalBatches) {
  if (limitConfig === null || limitConfig === undefined) return totalBatches;
  if (typeof limitConfig === 'number') {
    if (limitConfig > 0 && limitConfig <= 1) {
      return Math.max(1, Math.round(limitConfig * totalBatches));
    }
    return Math.min(limitConfig, totalBatches);
  }
  return totalBatches;
}

export async function noGradAsync(fn) {
  const { GradMode } = await import('../../../autograd/grad_mode.js');
  const prev = GradMode.isEnabled();
  GradMode.setEnabled(false);
  try {
    await fn();
  } finally {
    GradMode.setEnabled(prev);
  }
}
