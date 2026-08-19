import { describe, it, expect } from 'vitest';
import { GraphCycles } from '../../../../src/compiler/passes/fusion/graph_cycles.js';
import { scalingExponent, scalingReport, SUBQUADRATIC_EXPONENT } from '../../../_utils/scaling.js';

function mergeAll(pairs, n, edges) {
  const g = new GraphCycles(n, edges);
  for (const [x, y] of pairs) {
    const a = g.find(x);
    const b = g.find(y);
    if (a !== b && !g.wouldCreateCycle(a, b)) g.merge(a, b);
  }
  return g;
}

describe('GraphCycles keeps a valid topological order under contraction', () => {
  function ranksOf(g, nodes) {
    return nodes.map((i) => g._rank[g.find(i)]);
  }

  it('every edge still points forward after a chain is merged pairwise', () => {
    const n = 60;
    const edges = [];
    for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
    const g = mergeAll(Array.from({ length: n - 1 }, (_, i) => [i, i + 1]), n, edges);
    for (const [u, v] of edges) {
      const ru = g.find(u);
      const rv = g.find(v);
      if (ru === rv) continue;
      expect(g._rank[ru], `edge ${u} -> ${v} must point forward`).toBeLessThan(g._rank[rv]);
    }
  });

  it('contracting two unrelated nodes keeps every other edge ordered', () => {
    const n = 40;
    const edges = [];
    for (let i = 0; i + 1 < 10; i++) edges.push([i, i + 1]);
    for (let i = 20; i + 1 < 30; i++) edges.push([i, i + 1]);
    const g = mergeAll([[12, 35]], n, edges);
    for (const [u, v] of edges) {
      const ru = g.find(u);
      const rv = g.find(v);
      if (ru === rv) continue;
      expect(g._rank[ru], `edge ${u} -> ${v} must point forward`).toBeLessThan(g._rank[rv]);
    }
  });

  it('contracting across a populated window still orders the predecessors first', () => {
    const n = 30;
    const edges = [];
    for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
    const g = new GraphCycles(n, edges);
    expect(g.wouldCreateCycle(g.find(0), g.find(n - 1))).toBe(true);
  });

  it('a merged group reports the same root for both members', () => {
    const g = mergeAll([[3, 17]], 20, []);
    expect(g.find(3)).toBe(g.find(17));
    expect(ranksOf(g, [3, 17])[0]).toBe(ranksOf(g, [3, 17])[1]);
  });
});

describe('contraction cost tracks the nodes that actually have to move', () => {
  it('merging nodes with nothing between them stays sub-quadratic', () => {
    const measured = scalingExponent({
      build: (n) => ({ n, pairs: Array.from({ length: n >> 1 }, (_, k) => [k, n - 1 - k]) }),
      work: ({ n, pairs }) => mergeAll(pairs, n, []),
      n: 500,
    });
    expect(measured.exponent, scalingReport(measured)).toBeLessThan(SUBQUADRATIC_EXPONENT);
  });

  it('merging along a chain stays sub-quadratic', () => {
    const measured = scalingExponent({
      build: (n) => {
        const edges = [];
        for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
        return { n, edges, pairs: Array.from({ length: n - 1 }, (_, i) => [i, i + 1]) };
      },
      work: ({ n, edges, pairs }) => mergeAll(pairs, n, edges),
      n: 1000,
    });
    expect(measured.exponent, scalingReport(measured)).toBeLessThan(SUBQUADRATIC_EXPONENT);
  });
});
