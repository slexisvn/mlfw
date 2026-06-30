import { describe, it, expect } from 'vitest';
import { GraphCycles } from '../../../src/compiler/passes/fusion/graph_cycles.js';

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

class Oracle {
  constructor(n, edges) {
    this.n = n;
    this.edges = edges;
    this.parent = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }
  union(a, b) {
    this.parent[this.find(a)] = this.find(b);
  }
  _cycleWith(mapNode) {
    const adj = new Map();
    const nodes = new Set();
    for (let i = 0; i < this.n; i++) nodes.add(mapNode(i));
    const indeg = new Map();
    for (const g of nodes) {
      indeg.set(g, 0);
      adj.set(g, new Set());
    }
    for (const [u, v] of this.edges) {
      const gu = mapNode(u);
      const gv = mapNode(v);
      if (gu === gv) continue;
      if (!adj.get(gu).has(gv)) {
        adj.get(gu).add(gv);
        indeg.set(gv, indeg.get(gv) + 1);
      }
    }
    const queue = [];
    for (const g of nodes) if (indeg.get(g) === 0) queue.push(g);
    let seen = 0;
    let qi = 0;
    while (qi < queue.length) {
      const u = queue[qi++];
      seen++;
      for (const v of adj.get(u)) {
        const d = indeg.get(v) - 1;
        indeg.set(v, d);
        if (d === 0) queue.push(v);
      }
    }
    return seen < nodes.size;
  }
  wouldCreateCycle(a, b) {
    const ga = this.find(a);
    const gb = this.find(b);
    if (ga === gb) return false;
    return this._cycleWith((i) => {
      const g = this.find(i);
      return g === gb ? ga : g;
    });
  }
}

function randomDag(rng, n, edgeProb) {
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (rng() < edgeProb) edges.push([i, j]);
    }
  }
  return edges;
}

describe('GraphCycles — matches brute-force reachability oracle', () => {
  it('handcrafted diamond rejects head/tail contraction, allows siblings', () => {
    const gc = new GraphCycles(4, [[0, 1], [0, 2], [1, 3], [2, 3]]);
    expect(gc.wouldCreateCycle(0, 3)).toBe(true);
    expect(gc.wouldCreateCycle(1, 2)).toBe(false);
    expect(gc.wouldCreateCycle(0, 1)).toBe(false);
    gc.merge(1, 2);
    expect(gc.wouldCreateCycle(0, 3)).toBe(true);
  });

  it('agrees with oracle across random DAGs and incremental merges', () => {
    for (let trial = 0; trial < 40; trial++) {
      const rng = makeRng(1000 + trial * 7);
      const n = 6 + Math.floor(rng() * 18);
      const edges = randomDag(rng, n, 0.18 + rng() * 0.25);
      const gc = new GraphCycles(n, edges);
      const oracle = new Oracle(n, edges);

      for (let step = 0; step < n * 4; step++) {
        const a = Math.floor(rng() * n);
        const b = Math.floor(rng() * n);
        if (oracle.find(a) === oracle.find(b)) continue;

        const got = gc.wouldCreateCycle(a, b);
        const want = oracle.wouldCreateCycle(a, b);
        expect(got).toBe(want);

        if (!want && rng() < 0.7) {
          gc.merge(a, b);
          oracle.union(a, b);
          expect(gc.find(a)).toBe(gc.find(b));
        }
      }
    }
  });

  it('never lets a committed merge sequence form a cycle (find stays consistent)', () => {
    const rng = makeRng(424242);
    const n = 30;
    const edges = randomDag(rng, n, 0.2);
    const gc = new GraphCycles(n, edges);
    const oracle = new Oracle(n, edges);

    for (let step = 0; step < 500; step++) {
      const a = Math.floor(rng() * n);
      const b = Math.floor(rng() * n);
      if (gc.find(a) === gc.find(b)) continue;
      if (gc.wouldCreateCycle(a, b)) {
        expect(oracle.wouldCreateCycle(a, b)).toBe(true);
        continue;
      }
      expect(oracle.wouldCreateCycle(a, b)).toBe(false);
      gc.merge(a, b);
      oracle.union(a, b);
    }
  });
});
