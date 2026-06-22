function leafValue(rows, Y) {
  if (rows.length === 0) return 0;
  let sum = 0;
  for (const i of rows) sum += Y[i];
  return sum / rows.length;
}

function buildTree(X, Y, rows, depth, maxDepth, minSamples) {
  const n = rows.length;
  if (depth >= maxDepth || n < minSamples * 2) {
    return { leaf: leafValue(rows, Y) };
  }
  const dim = X[0].length;
  let totalSum = 0;
  let totalSumSq = 0;
  for (const i of rows) { const y = Y[i]; totalSum += y; totalSumSq += y * y; }

  let best = null;
  let bestSorted = null;
  for (let f = 0; f < dim; f++) {
    const sorted = rows.slice().sort((a, b) => X[a][f] - X[b][f]);
    let leftSum = 0;
    let leftSumSq = 0;
    for (let s = 0; s < sorted.length - 1; s++) {
      const y = Y[sorted[s]];
      leftSum += y;
      leftSumSq += y * y;
      const xa = X[sorted[s]][f];
      const xb = X[sorted[s + 1]][f];
      if (xa === xb) continue;
      const leftCount = s + 1;
      const rightCount = n - leftCount;
      if (leftCount < minSamples || rightCount < minSamples) continue;
      const rightSum = totalSum - leftSum;
      const rightSumSq = totalSumSq - leftSumSq;
      const cost = (leftSumSq - (leftSum * leftSum) / leftCount) + (rightSumSq - (rightSum * rightSum) / rightCount);
      if (!best || cost < best.cost) { best = { f, thr: (xa + xb) / 2, sIdx: s, cost }; bestSorted = sorted; }
    }
  }
  if (!best) return { leaf: leafValue(rows, Y) };

  const sorted = bestSorted;
  return {
    f: best.f, thr: best.thr,
    left: buildTree(X, Y, sorted.slice(0, best.sIdx + 1), depth + 1, maxDepth, minSamples),
    right: buildTree(X, Y, sorted.slice(best.sIdx + 1), depth + 1, maxDepth, minSamples)
  };
}

function predictTree(node, x) {
  while (node && node.leaf === undefined) {
    node = x[node.f] <= node.thr ? node.left : node.right;
  }
  return node ? node.leaf : 0;
}

export class GradientBoostedTrees {
  constructor(opts = {}) {
    this.numTrees = opts.numTrees ?? 60;
    this.maxDepth = opts.maxDepth ?? 3;
    this.lr = opts.lr ?? 0.1;
    this.minSamples = opts.minSamples ?? 1;
    this.trees = null;
    this.base = 0;
  }

  fit(X, Y) {
    const n = X.length;
    if (n === 0) return;
    let base = 0;
    for (const y of Y) base += y;
    base /= n;
    const preds = new Array(n).fill(base);
    const rows = [];
    for (let i = 0; i < n; i++) rows.push(i);
    const trees = [];
    for (let t = 0; t < this.numTrees; t++) {
      const residuals = new Array(n);
      for (let i = 0; i < n; i++) residuals[i] = Y[i] - preds[i];
      const tree = buildTree(X, residuals, rows, 0, this.maxDepth, this.minSamples);
      for (let i = 0; i < n; i++) preds[i] += this.lr * predictTree(tree, X[i]);
      trees.push(tree);
    }
    this.trees = trees;
    this.base = base;
  }

  predict(x) {
    if (!this.trees) return 0;
    let p = this.base;
    for (const tree of this.trees) p += this.lr * predictTree(tree, x);
    return p;
  }

  serialize() {
    return { trees: this.trees, base: this.base, lr: this.lr };
  }

  static deserialize(data) {
    const g = new GradientBoostedTrees({ lr: data.lr });
    g.trees = data.trees;
    g.base = data.base;
    return g;
  }
}
