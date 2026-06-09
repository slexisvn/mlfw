export function clipGradNorm_(parameters, maxNorm, normType = 2) {
  const params = Array.isArray(parameters) ? parameters : [...parameters];
  let totalNorm = 0;

  if (normType === Infinity) {
    for (const p of params) {
      if (p.grad === null) continue;
      const d = p.grad._impl.storage.data;
      for (let i = 0; i < d.length; i++) {
        const a = Math.abs(d[i]);
        if (a > totalNorm) totalNorm = a;
      }
    }
  } else if (normType === 2) {
    for (const p of params) {
      if (p.grad === null) continue;
      const d = p.grad._impl.storage.data;
      for (let i = 0; i < d.length; i++) totalNorm += d[i] * d[i];
    }
    totalNorm = Math.sqrt(totalNorm);
  } else {
    for (const p of params) {
      if (p.grad === null) continue;
      const d = p.grad._impl.storage.data;
      for (let i = 0; i < d.length; i++) totalNorm += Math.pow(Math.abs(d[i]), normType);
    }
    totalNorm = Math.pow(totalNorm, 1 / normType);
  }

  const clipCoef = maxNorm / (totalNorm + 1e-6);
  if (clipCoef < 1) {
    for (const p of params) {
      if (p.grad === null) continue;
      const d = p.grad._impl.storage.data;
      for (let i = 0; i < d.length; i++) d[i] *= clipCoef;
      p.grad._impl.bumpVersion();
    }
  }

  return totalNorm;
}

export function clipGradValue_(parameters, clipValue) {
  const params = Array.isArray(parameters) ? parameters : [...parameters];
  for (const p of params) {
    if (p.grad === null) continue;
    const d = p.grad._impl.storage.data;
    for (let i = 0; i < d.length; i++) {
      if (d[i] > clipValue) d[i] = clipValue;
      else if (d[i] < -clipValue) d[i] = -clipValue;
    }
    p.grad._impl.bumpVersion();
  }
}
