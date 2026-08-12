import { hostMatrix, hostColumns, toHostTensor } from '../../../tensor/utils/host_matrix.js';
import { encodeLabels } from '../../../ml/_util.js';
import { makeRng } from '../../../ml/_random.js';
import type { Rng } from '../../../ml/types.js';
import type { Tensor } from '../../../tensor/core/tensor.js';

type Split = { feature: number; threshold: number; impurity: number };

function chooseFeatures(d: number, maxFeatures: number, rng: Rng): number[] {
  if (maxFeatures <= 0 || maxFeatures >= d) return Array.from({ length: d }, (_, i) => i);
  const pool = Array.from({ length: d }, (_, i) => i);
  for (let i = d - 1; i > d - 1 - maxFeatures; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
  }
  return pool.slice(d - maxFeatures);
}

function bestSplitClassify(x: Float64Array, d: number, indices: readonly number[], feats: readonly number[], K: number, labels: Int32Array, minLeaf: number): Split | null {
  const m = indices.length;
  let best: Split | null = null;
  for (const f of feats) {
    const order = indices.slice().sort((a, b) => x[a * d + f] - x[b * d + f]);
    const leftCounts = new Float64Array(K);
    const rightCounts = new Float64Array(K);
    for (const i of order) rightCounts[labels[i]]++;
    let leftN = 0;
    for (let s = 0; s < m - 1; s++) {
      const lbl = labels[order[s]];
      leftCounts[lbl]++;
      rightCounts[lbl]--;
      leftN++;
      const rightN = m - leftN;
      const vx = x[order[s] * d + f];
      const nx = x[order[s + 1] * d + f];
      if (vx === nx || leftN < minLeaf || rightN < minLeaf) continue;
      let giniL = 1;
      let giniR = 1;
      for (let c = 0; c < K; c++) {
        const pl = leftCounts[c] / leftN;
        const pr = rightCounts[c] / rightN;
        giniL -= pl * pl;
        giniR -= pr * pr;
      }
      const impurity = (leftN * giniL + rightN * giniR) / m;
      if (!best || impurity < best.impurity) best = { feature: f, threshold: (vx + nx) / 2, impurity };
    }
  }
  return best;
}

function bestSplitRegress(x: Float64Array, d: number, indices: readonly number[], feats: readonly number[], yData: Float64Array, minLeaf: number): Split | null {
  const m = indices.length;
  let best: Split | null = null;
  for (const f of feats) {
    const order = indices.slice().sort((a, b) => x[a * d + f] - x[b * d + f]);
    let totalSum = 0;
    let totalSq = 0;
    for (const i of order) { totalSum += yData[i]; totalSq += yData[i] * yData[i]; }
    let leftSum = 0;
    let leftSq = 0;
    let leftN = 0;
    for (let s = 0; s < m - 1; s++) {
      const yv = yData[order[s]];
      leftSum += yv;
      leftSq += yv * yv;
      leftN++;
      const rightN = m - leftN;
      const vx = x[order[s] * d + f];
      const nx = x[order[s + 1] * d + f];
      if (vx === nx || leftN < minLeaf || rightN < minLeaf) continue;
      const sseL = leftSq - (leftSum * leftSum) / leftN;
      const rightSum = totalSum - leftSum;
      const rightSq = totalSq - leftSq;
      const sseR = rightSq - (rightSum * rightSum) / rightN;
      const impurity = sseL + sseR;
      if (!best || impurity < best.impurity) best = { feature: f, threshold: (vx + nx) / 2, impurity };
    }
  }
  return best;
}

export function cpuDecisionTreeFit(_keySet: unknown, X: Tensor, y: Tensor, maxDepth: number, minSplit: number, minLeaf: number, maxFeatures: number, classify: boolean, seed: number): Tensor[] {
  const m = hostMatrix(X);
  const x = m.data;
  const d = m.cols;
  const yv = hostColumns(y);
  const rng = makeRng(seed);

  let labels: Int32Array | null = null;
  let classes: number[] | null = null;
  let K = 0;
  if (classify) {
    const enc = encodeLabels(yv.data, yv.rows);
    labels = enc.y;
    classes = enc.classes;
    K = classes.length;
  }

  const feature: number[] = [];
  const threshold: number[] = [];
  const left: number[] = [];
  const right: number[] = [];
  const value: number[] = [];

  const leafValue = (indices: readonly number[]): number => {
    if (classify) {
      const counts = new Float64Array(K);
      for (const i of indices) counts[(labels as Int32Array)[i]]++;
      let best = 0;
      for (let c = 1; c < K; c++) if (counts[c] > counts[best]) best = c;
      return (classes as number[])[best];
    }
    let s = 0;
    for (const i of indices) s += yv.data[i];
    return s / indices.length;
  };

  const isPure = (indices: readonly number[]): boolean => {
    const ref = (classify ? labels : yv.data) as Int32Array | Float64Array;
    const first = ref[indices[0]];
    for (const i of indices) if (ref[i] !== first) return false;
    return true;
  };

  const build = (indices: number[], depth: number): number => {
    const nodeId = feature.length;
    feature.push(-1); threshold.push(0); left.push(-1); right.push(-1); value.push(0);

    if (depth >= maxDepth || indices.length < minSplit || isPure(indices)) {
      value[nodeId] = leafValue(indices);
      return nodeId;
    }
    const feats = chooseFeatures(d, maxFeatures, rng);
    const split = classify
      ? bestSplitClassify(x, d, indices, feats, K, labels as Int32Array, minLeaf)
      : bestSplitRegress(x, d, indices, feats, yv.data, minLeaf);
    if (!split) {
      value[nodeId] = leafValue(indices);
      return nodeId;
    }
    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    for (const i of indices) {
      if (x[i * d + split.feature] <= split.threshold) leftIdx.push(i);
      else rightIdx.push(i);
    }
    feature[nodeId] = split.feature;
    threshold[nodeId] = split.threshold;
    left[nodeId] = build(leftIdx, depth + 1);
    right[nodeId] = build(rightIdx, depth + 1);
    return nodeId;
  };

  build(Array.from({ length: m.rows }, (_, i) => i), 0);

  return [
    toHostTensor(Float64Array.from(feature), [feature.length], X.dtype, X.device),
    toHostTensor(Float64Array.from(threshold), [threshold.length], X.dtype, X.device),
    toHostTensor(Float64Array.from(left), [left.length], X.dtype, X.device),
    toHostTensor(Float64Array.from(right), [right.length], X.dtype, X.device),
    toHostTensor(Float64Array.from(value), [value.length], X.dtype, X.device),
  ];
}

export function cpuDecisionTreePredict(_keySet: unknown, X: Tensor, featureT: Tensor, thresholdT: Tensor, leftT: Tensor, rightT: Tensor, valueT: Tensor): Tensor {
  const m = hostMatrix(X);
  const feature = hostColumns(featureT).data;
  const threshold = hostColumns(thresholdT).data;
  const left = hostColumns(leftT).data;
  const right = hostColumns(rightT).data;
  const value = hostColumns(valueT).data;
  const out = new Float64Array(m.rows);
  const d = m.cols;
  for (let i = 0; i < m.rows; i++) {
    let node = 0;
    while (feature[node] !== -1) {
      if (m.data[i * d + feature[node]] <= threshold[node]) node = left[node];
      else node = right[node];
    }
    out[i] = value[node];
  }
  return toHostTensor(out, [m.rows], X.dtype, X.device);
}
