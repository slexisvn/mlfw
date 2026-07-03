export function qrHost(a, m, n) {
  const k = Math.min(m, n);
  const R = Float64Array.from(a);
  const vs = [];
  for (let j = 0; j < k; j++) {
    let normx = 0;
    for (let i = j; i < m; i++) normx += R[i * n + j] * R[i * n + j];
    normx = Math.sqrt(normx);
    if (normx === 0) {
      vs.push(null);
      continue;
    }
    const alpha = R[j * n + j] >= 0 ? -normx : normx;
    const v = new Float64Array(m - j);
    v[0] = R[j * n + j] - alpha;
    for (let i = j + 1; i < m; i++) v[i - j] = R[i * n + j];
    let vnorm2 = 0;
    for (let i = 0; i < v.length; i++) vnorm2 += v[i] * v[i];
    if (vnorm2 === 0) {
      vs.push(null);
      continue;
    }
    vs.push(v);
    for (let c = j; c < n; c++) {
      let dotv = 0;
      for (let i = 0; i < v.length; i++) dotv += v[i] * R[(j + i) * n + c];
      const s = (2 * dotv) / vnorm2;
      for (let i = 0; i < v.length; i++) R[(j + i) * n + c] -= s * v[i];
    }
  }

  const Q = new Float64Array(m * k);
  for (let c = 0; c < k; c++) Q[c * k + c] = 1;
  for (let j = k - 1; j >= 0; j--) {
    const v = vs[j];
    if (!v) continue;
    let vnorm2 = 0;
    for (let i = 0; i < v.length; i++) vnorm2 += v[i] * v[i];
    for (let c = 0; c < k; c++) {
      let dotv = 0;
      for (let i = 0; i < v.length; i++) dotv += v[i] * Q[(j + i) * k + c];
      const s = (2 * dotv) / vnorm2;
      for (let i = 0; i < v.length; i++) Q[(j + i) * k + c] -= s * v[i];
    }
  }

  const Rout = new Float64Array(k * n);
  for (let i = 0; i < k; i++) for (let j = i; j < n; j++) Rout[i * n + j] = R[i * n + j];
  return { Q, R: Rout, k };
}
