import { describe, it, expect } from 'vitest';
import { rangesOverlap } from '../../../src/compiler/schedule/dep_analysis.js';

describe('BufferRegion overlap (rangesOverlap)', () => {
  it('disjoint 1-D slices of the same buffer do not overlap', () => {
    expect(rangesOverlap([[0, 4]], [[4, 4]])).toBe(false);
    expect(rangesOverlap([[0, 4]], [[8, 4]])).toBe(false);
  });

  it('overlapping 1-D slices overlap', () => {
    expect(rangesOverlap([[0, 4]], [[2, 4]])).toBe(true);
    expect(rangesOverlap([[0, 8]], [[0, 8]])).toBe(true);
  });

  it('disjoint in any dimension => regions do not overlap', () => {
    expect(rangesOverlap([[0, 2], [0, 2]], [[0, 2], [2, 2]])).toBe(false);
    expect(rangesOverlap([[0, 2], [0, 2]], [[2, 2], [0, 2]])).toBe(false);
  });

  it('overlap in all dimensions => overlap', () => {
    expect(rangesOverlap([[0, 4], [0, 4]], [[2, 4], [1, 2]])).toBe(true);
  });

  it('unknown ranges are treated conservatively as overlapping', () => {
    expect(rangesOverlap(null, [[0, 4]])).toBe(true);
    expect(rangesOverlap([[0, 4]], null)).toBe(true);
    expect(rangesOverlap([null], [[0, 4]])).toBe(true);
    expect(rangesOverlap([[0, 2], null], [[0, 2], [8, 2]])).toBe(true);
  });
});
