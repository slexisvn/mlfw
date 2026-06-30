import { describe, it, expect } from 'vitest';
import { MaxHeap } from '../../../src/compiler/passes/fusion/binary_heap.js';

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

describe('MaxHeap', () => {
  it('pops in descending priority order', () => {
    const rng = makeRng(7);
    for (let trial = 0; trial < 50; trial++) {
      const heap = new MaxHeap();
      const n = Math.floor(rng() * 200);
      const priorities = [];
      for (let i = 0; i < n; i++) {
        const p = Math.floor(rng() * 1000) - 500;
        priorities.push(p);
        heap.push(p, { p });
      }
      expect(heap.size).toBe(n);
      priorities.sort((a, b) => b - a);
      const popped = [];
      while (!heap.isEmpty()) popped.push(heap.pop().p);
      expect(popped).toEqual(priorities);
    }
  });

  it('interleaves pushes and pops while keeping the max on top', () => {
    const rng = makeRng(99);
    const heap = new MaxHeap();
    const live = [];
    for (let i = 0; i < 2000; i++) {
      if (rng() < 0.6 || live.length === 0) {
        const p = Math.floor(rng() * 10000);
        heap.push(p, p);
        live.push(p);
      } else {
        const got = heap.pop();
        const want = Math.max(...live);
        live.splice(live.indexOf(want), 1);
        expect(got).toBe(want);
      }
    }
  });

  it('returns undefined when empty', () => {
    const heap = new MaxHeap();
    expect(heap.pop()).toBe(undefined);
    expect(heap.isEmpty()).toBe(true);
  });
});
