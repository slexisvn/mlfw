export const numel = (shape) => shape.reduce((a, b) => a * b, 1);

export const strides = (shape) => {
  const s = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) s[i] = s[i + 1] * shape[i + 1];
  return s;
};

export const at = (data, shape, idx) => {
  const st = strides(shape);
  let off = 0;
  for (let i = 0; i < idx.length; i++) off += idx[i] * st[i];
  return data[off];
};

export const map1 = (a, f) => Float32Array.from(a, f);
export const map2 = (a, b, f) => Float32Array.from(a, (v, i) => f(v, b[i]));

export const add = (a, b) => map2(a, b, (x, y) => x + y);
export const sub = (a, b) => map2(a, b, (x, y) => x - y);
export const mul = (a, b) => map2(a, b, (x, y) => x * y);
export const div = (a, b) => map2(a, b, (x, y) => x / y);

export const relu = (a) => map1(a, (x) => Math.max(0, x));
export const sigmoid = (a) => map1(a, (x) => 1 / (1 + Math.exp(-x)));
export const tanh = (a) => map1(a, Math.tanh);
export const exp = (a) => map1(a, Math.exp);
export const sqrt = (a) => map1(a, Math.sqrt);
export const GELU_SIGMOID_COEFF = 1.702;
export const gelu = (a) => map1(a, (x) => x / (1 + Math.exp(-GELU_SIGMOID_COEFF * x)));
export const silu = (a) => map1(a, (x) => x / (1 + Math.exp(-x)));

export function matmul(a, [m, k], b, [k2, n]) {
  if (k !== k2) throw new Error(`matmul shape mismatch: ${k} vs ${k2}`);
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
      out[i * n + j] = acc;
    }
  }
  return out;
}

export function batchMatmul(a, aShape, b, bShape) {
  const [m, k] = aShape.slice(-2);
  const [, n] = bShape.slice(-2);
  const batch = numel(aShape.slice(0, -2));
  const out = new Float32Array(batch * m * n);
  for (let bi = 0; bi < batch; bi++) {
    const sub = matmul(a.subarray(bi * m * k, (bi + 1) * m * k), [m, k], b.subarray(bi * k * n, (bi + 1) * k * n), [k, n]);
    out.set(sub, bi * m * n);
  }
  return out;
}

export function transpose(a, shape, perm) {
  const outShape = perm.map((p) => shape[p]);
  const out = new Float32Array(numel(shape));
  const outStrides = strides(outShape);
  const inStrides = strides(shape);
  const idx = new Array(shape.length).fill(0);
  for (let flat = 0; flat < out.length; flat++) {
    let rem = flat;
    for (let d = 0; d < outShape.length; d++) { idx[d] = Math.floor(rem / outStrides[d]); rem -= idx[d] * outStrides[d]; }
    let inOff = 0;
    for (let d = 0; d < perm.length; d++) inOff += idx[d] * inStrides[perm[d]];
    out[flat] = a[inOff];
  }
  return out;
}

export function slice(a, shape, starts, limits) {
  const outShape = starts.map((s, i) => limits[i] - s);
  const out = new Float32Array(numel(outShape));
  const inStrides = strides(shape);
  const outStrides = strides(outShape);
  const idx = new Array(shape.length).fill(0);
  for (let flat = 0; flat < out.length; flat++) {
    let rem = flat;
    for (let d = 0; d < outShape.length; d++) { idx[d] = Math.floor(rem / outStrides[d]); rem -= idx[d] * outStrides[d]; }
    let inOff = 0;
    for (let d = 0; d < shape.length; d++) inOff += (idx[d] + starts[d]) * inStrides[d];
    out[flat] = a[inOff];
  }
  return out;
}

export function concat(arrays, shapes, axis) {
  const outShape = shapes[0].slice();
  outShape[axis] = shapes.reduce((s, sh) => s + sh[axis], 0);
  const out = new Float32Array(numel(outShape));
  const outStrides = strides(outShape);
  let offsetAlongAxis = 0;
  for (let t = 0; t < arrays.length; t++) {
    const sh = shapes[t];
    const inStrides = strides(sh);
    const idx = new Array(sh.length).fill(0);
    for (let flat = 0; flat < numel(sh); flat++) {
      let rem = flat;
      for (let d = 0; d < sh.length; d++) { idx[d] = Math.floor(rem / inStrides[d]); rem -= idx[d] * inStrides[d]; }
      let outOff = 0;
      for (let d = 0; d < sh.length; d++) outOff += (d === axis ? idx[d] + offsetAlongAxis : idx[d]) * outStrides[d];
      out[outOff] = arrays[t][flat];
    }
    offsetAlongAxis += sh[axis];
  }
  return out;
}

export function broadcastTo(a, shape, outShape) {
  const pad = outShape.length - shape.length;
  const full = new Array(pad).fill(1).concat(shape);
  const out = new Float32Array(numel(outShape));
  const outStrides = strides(outShape);
  const inStrides = strides(full);
  const idx = new Array(outShape.length).fill(0);
  for (let flat = 0; flat < out.length; flat++) {
    let rem = flat;
    for (let d = 0; d < outShape.length; d++) { idx[d] = Math.floor(rem / outStrides[d]); rem -= idx[d] * outStrides[d]; }
    let inOff = 0;
    for (let d = 0; d < outShape.length; d++) inOff += (full[d] === 1 ? 0 : idx[d]) * inStrides[d];
    out[flat] = a[inOff];
  }
  return out;
}

export function reduce(a, shape, axes, kind) {
  const keep = shape.map((d, i) => (axes.includes(i) ? 1 : d));
  const outShape = shape.filter((_, i) => !axes.includes(i));
  if (kind === 'mean') {
    const count = axes.reduce((a, d) => a * shape[d], 1);
    return map1(reduce(a, shape, axes, 'sum'), (v) => v / count);
  }
  const init = kind === 'max' ? -Infinity : kind === 'min' ? Infinity : kind === 'prod' ? 1 : 0;
  const out = new Float32Array(numel(outShape) || 1).fill(init);
  const inStrides = strides(shape);
  const keepStrides = strides(keep);
  const idx = new Array(shape.length).fill(0);
  const outIdxOf = (i) => {
    let off = 0;
    let k = 0;
    const os = strides(outShape);
    for (let d = 0; d < shape.length; d++) if (!axes.includes(d)) off += i[d] * os[k++];
    return off;
  };
  for (let flat = 0; flat < numel(shape); flat++) {
    let rem = flat;
    for (let d = 0; d < shape.length; d++) { idx[d] = Math.floor(rem / inStrides[d]); rem -= idx[d] * inStrides[d]; }
    const o = outIdxOf(idx);
    const v = a[flat];
    if (kind === 'sum') out[o] += v;
    else if (kind === 'prod') out[o] *= v;
    else if (kind === 'max') out[o] = Math.max(out[o], v);
    else if (kind === 'min') out[o] = Math.min(out[o], v);
    else throw new Error(`unknown reduce kind '${kind}'`);
  }
  void keepStrides;
  return out;
}

export function softmaxLastAxis(a, shape) {
  const n = shape[shape.length - 1];
  const rows = numel(shape) / n;
  const out = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let i = 0; i < n; i++) max = Math.max(max, a[r * n + i]);
    let sum = 0;
    for (let i = 0; i < n; i++) { const e = Math.exp(a[r * n + i] - max); out[r * n + i] = e; sum += e; }
    for (let i = 0; i < n; i++) out[r * n + i] /= sum;
  }
  return out;
}

export function layerNormLastAxis(a, shape, { gamma = null, beta = null, eps = 1e-5 } = {}) {
  const n = shape[shape.length - 1];
  const rows = numel(shape) / n;
  const out = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += a[r * n + i];
    mean /= n;
    let varr = 0;
    for (let i = 0; i < n; i++) { const d = a[r * n + i] - mean; varr += d * d; }
    varr /= n;
    const inv = 1 / Math.sqrt(varr + eps);
    for (let i = 0; i < n; i++) {
      const norm = (a[r * n + i] - mean) * inv;
      out[r * n + i] = norm * (gamma ? gamma[i] : 1) + (beta ? beta[i] : 0);
    }
  }
  return out;
}

export function resizeNearest(a, [N, C, H, W], [OH, OW]) {
  const out = new Float32Array(N * C * OH * OW);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let oh = 0; oh < OH; oh++) {
        const ih = Math.min(H - 1, Math.floor((oh * H) / OH));
        for (let ow = 0; ow < OW; ow++) {
          const iw = Math.min(W - 1, Math.floor((ow * W) / OW));
          out[((n * C + c) * OH + oh) * OW + ow] = a[((n * C + c) * H + ih) * W + iw];
        }
      }
    }
  }
  return out;
}

export function conv2d(x, [N, C, H, W], w, [F, , KH, KW], { stride = 1, pad = 0, dilation = 1, groups = 1 } = {}) {
  const OH = Math.floor((H + 2 * pad - dilation * (KH - 1) - 1) / stride) + 1;
  const OW = Math.floor((W + 2 * pad - dilation * (KW - 1) - 1) / stride) + 1;
  const out = new Float32Array(N * F * OH * OW);
  const cPerGroup = C / groups;
  const fPerGroup = F / groups;
  for (let n = 0; n < N; n++) {
    for (let f = 0; f < F; f++) {
      const g = Math.floor(f / fPerGroup);
      for (let oh = 0; oh < OH; oh++) {
        for (let ow = 0; ow < OW; ow++) {
          let acc = 0;
          for (let ci = 0; ci < cPerGroup; ci++) {
            const c = g * cPerGroup + ci;
            for (let kh = 0; kh < KH; kh++) {
              for (let kw = 0; kw < KW; kw++) {
                const ih = oh * stride - pad + kh * dilation;
                const iw = ow * stride - pad + kw * dilation;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                acc += x[((n * C + c) * H + ih) * W + iw] * w[((f * cPerGroup + ci) * KH + kh) * KW + kw];
              }
            }
          }
          out[((n * F + f) * OH + oh) * OW + ow] = acc;
        }
      }
    }
  }
  return { data: out, shape: [N, F, OH, OW] };
}

export function pool2d(x, [N, C, H, W], kind, [KH, KW], [SH, SW] = [KH, KW], pad = 0) {
  const OH = Math.floor((H + 2 * pad - KH) / SH) + 1;
  const OW = Math.floor((W + 2 * pad - KW) / SW) + 1;
  const out = new Float32Array(N * C * OH * OW);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let oh = 0; oh < OH; oh++) {
        for (let ow = 0; ow < OW; ow++) {
          let acc = kind === 'max' ? -Infinity : 0;
          let count = 0;
          for (let kh = 0; kh < KH; kh++) {
            for (let kw = 0; kw < KW; kw++) {
              const ih = oh * SH - pad + kh;
              const iw = ow * SW - pad + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              const v = x[((n * C + c) * H + ih) * W + iw];
              acc = kind === 'max' ? Math.max(acc, v) : acc + v;
              count++;
            }
          }
          out[((n * C + c) * OH + oh) * OW + ow] = kind === 'max' ? acc : acc / count;
        }
      }
    }
  }
  return { data: out, shape: [N, C, OH, OW] };
}

export function batchNorm(x, shape, gamma, beta, mean, variance, { axis = 1, eps = 1e-5 } = {}) {
  const out = new Float32Array(x.length);
  const st = strides(shape);
  const idx = new Array(shape.length).fill(0);
  for (let flat = 0; flat < x.length; flat++) {
    let rem = flat;
    for (let d = 0; d < shape.length; d++) { idx[d] = Math.floor(rem / st[d]); rem -= idx[d] * st[d]; }
    const c = idx[axis];
    out[flat] = ((x[flat] - mean[c]) / Math.sqrt(variance[c] + eps)) * gamma[c] + beta[c];
  }
  return out;
}

export function transformerEncoderBlock(x, { seq, dModel, dFF, wq, wk, wv, wo, g1, b1, wff1, wff2, g2, b2, mask = null }) {
  const project = (w) => matmul(x, [seq, dModel], w, [dModel, dModel]);
  const kt = transpose(project(wk), [seq, dModel], [1, 0]);
  let scores = map1(matmul(project(wq), [seq, dModel], kt, [dModel, seq]), (s) => s / Math.sqrt(dModel));
  if (mask) scores = add(scores, mask);
  const ctx = matmul(softmaxLastAxis(scores, [seq, seq]), [seq, seq], project(wv), [seq, dModel]);
  const res1 = add(x, matmul(ctx, [seq, dModel], wo, [dModel, dModel]));
  const norm1 = layerNormLastAxis(res1, [seq, dModel], { gamma: g1, beta: b1 });
  const ff = gelu(matmul(norm1, [seq, dModel], wff1, [dModel, dFF]));
  const res2 = add(norm1, matmul(ff, [seq, dFF], wff2, [dFF, dModel]));
  return layerNormLastAxis(res2, [seq, dModel], { gamma: g2, beta: b2 });
}

export function expectClose(actual, expected, tol, label = '') {
  if (actual.length !== expected.length) {
    throw new Error(`${label} length ${actual.length} != expected ${expected.length}`);
  }
  let worstIdx = -1;
  let worstErr = 0;
  for (let i = 0; i < expected.length; i++) {
    const err = Math.abs(actual[i] - expected[i]) / (1 + Math.abs(expected[i]));
    if (err > worstErr) { worstErr = err; worstIdx = i; }
  }
  return {
    ok: worstErr <= tol,
    worstErr,
    worstIdx,
    message: worstIdx < 0 ? `${label} matches` : `${label} worst rel err ${worstErr.toExponential(2)} at index ${worstIdx}: got ${actual[worstIdx]}, expected ${expected[worstIdx]}`,
  };
}
