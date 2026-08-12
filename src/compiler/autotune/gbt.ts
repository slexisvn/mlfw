export type FeatureVector = readonly number[];
export type FeatureMatrix = readonly FeatureVector[];
export type LeafNode = { leaf: number };
export type SplitNode = { f: number; thr: number; left: TreeNode; right: TreeNode };
export type TreeNode = LeafNode | SplitNode;
export type SerializedGBT = { trees: TreeNode[] | null; base: number; lr: number };
type BestSplit = { f: number; thr: number; sIdx: number; cost: number };

function leafValue(rows: readonly number[], Y: readonly number[]): number {
  if (rows.length === 0) return 0;
  let sum = 0;
  for (const i of rows) sum += Y[i];
  return sum / rows.length;
}

function buildTree(X: FeatureMatrix, Y: readonly number[], rows: readonly number[], depth: number, maxDepth: number, minSamples: number, featureOrder: readonly number[][]): TreeNode {
  const n = rows.length;
  if (depth >= maxDepth || n < minSamples * 2) {
    return { leaf: leafValue(rows, Y) };
  }
  const dim = X[0].length;
  let totalSum = 0;
  let totalSumSq = 0;
  const rowSet = new Set(rows);
  for (const i of rows) { const y = Y[i]; totalSum += y; totalSumSq += y * y; }

  let best: BestSplit | null = null;
  let bestSorted: number[] | null = null;
  for (let f = 0; f < dim; f++) {
    const sorted = featureOrder[f].filter(r => rowSet.has(r));
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

  const sorted = bestSorted as number[];
  return {
    f: best.f, thr: best.thr,
    left: buildTree(X, Y, sorted.slice(0, best.sIdx + 1), depth + 1, maxDepth, minSamples, featureOrder),
    right: buildTree(X, Y, sorted.slice(best.sIdx + 1), depth + 1, maxDepth, minSamples, featureOrder)
  };
}

function buildFeatureOrder(X: FeatureMatrix): number[][] {
  const n = X.length;
  const dim = n > 0 ? X[0].length : 0;
  const order = new Array<number[]>(dim);
  const base: number[] = [];
  for (let i = 0; i < n; i++) base.push(i);
  for (let f = 0; f < dim; f++) {
    order[f] = base.slice().sort((a, b) => X[a][f] - X[b][f]);
  }
  return order;
}

function predictTree(node: TreeNode | null | undefined, x: FeatureVector): number {
  let cur = node;
  while (cur && (cur as LeafNode).leaf === undefined) {
    const sp = cur as SplitNode;
    cur = x[sp.f] <= sp.thr ? sp.left : sp.right;
  }
  return cur ? (cur as LeafNode).leaf : 0;
}

export class GradientBoostedTrees {
  numTrees: number;
  maxDepth: number;
  lr: number;
  minSamples: number;
  trees: TreeNode[] | null;
  base: number;

  constructor(opts: Readonly<{ numTrees?: number; maxDepth?: number; lr?: number; minSamples?: number }> = {}) {
    this.numTrees = opts.numTrees ?? 60;
    this.maxDepth = opts.maxDepth ?? 3;
    this.lr = opts.lr ?? 0.1;
    this.minSamples = opts.minSamples ?? 1;
    this.trees = null;
    this.base = 0;
  }

  fit(X: FeatureMatrix, Y: readonly number[]): void {
    const n = X.length;
    if (n === 0) return;
    let base = 0;
    for (const y of Y) base += y;
    base /= n;
    const preds: number[] = new Array<number>(n).fill(base);
    const rows: number[] = [];
    for (let i = 0; i < n; i++) rows.push(i);
    const featureOrder = buildFeatureOrder(X);
    const trees: TreeNode[] = [];
    for (let t = 0; t < this.numTrees; t++) {
      const residuals = new Array<number>(n);
      for (let i = 0; i < n; i++) residuals[i] = Y[i] - preds[i];
      const tree = buildTree(X, residuals, rows, 0, this.maxDepth, this.minSamples, featureOrder);
      for (let i = 0; i < n; i++) preds[i] += this.lr * predictTree(tree, X[i]);
      trees.push(tree);
    }
    this.trees = trees;
    this.base = base;
  }

  predict(x: FeatureVector): number {
    if (!this.trees) return 0;
    let p = this.base;
    for (const tree of this.trees) p += this.lr * predictTree(tree, x);
    return p;
  }

  serialize(): SerializedGBT {
    return { trees: this.trees, base: this.base, lr: this.lr };
  }

  static deserialize(data: SerializedGBT): GradientBoostedTrees {
    const g = new GradientBoostedTrees({ lr: data.lr });
    g.trees = data.trees;
    g.base = data.base;
    return g;
  }
}
