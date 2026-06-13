import { describe, it, expect } from 'vitest';
import { MemoryPool, MemoryBlock, BufferAssignment } from '../../../src/compiler/passes/memory/buffer_assignment.js';
import { BufferInterval } from '../../../src/compiler/passes/memory/buffer_liveness.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';

describe('MemoryBlock', () => {
  it('end equals offset + size', () => {
    const block = new MemoryBlock(64, 128, null);
    expect(block.end).toBe(192);
  });

  it('overlaps returns true for intersecting blocks', () => {
    const a = new MemoryBlock(0, 100, null);
    const b = new MemoryBlock(50, 100, null);
    expect(a.overlaps(b)).toBe(true);
    expect(b.overlaps(a)).toBe(true);
  });

  it('overlaps returns false for non-intersecting blocks', () => {
    const a = new MemoryBlock(0, 64, null);
    const b = new MemoryBlock(128, 64, null);
    expect(a.overlaps(b)).toBe(false);
  });

  it('overlaps returns false for adjacent blocks', () => {
    const a = new MemoryBlock(0, 64, null);
    const b = new MemoryBlock(64, 64, null);
    expect(a.overlaps(b)).toBe(false);
  });
});

describe('MemoryPool', () => {
  it('first allocation starts at offset 0', () => {
    const pool = new MemoryPool('global', 64);
    const buf = new Buffer('a', [16], 'f32', 'global');
    const block = pool.allocate(64, buf);
    expect(block.offset).toBe(0);
  });

  it('second allocation does not overlap first', () => {
    const pool = new MemoryPool('global', 64);
    const buf1 = new Buffer('a', [16], 'f32', 'global');
    const buf2 = new Buffer('b', [16], 'f32', 'global');
    const b1 = pool.allocate(64, buf1);
    const b2 = pool.allocate(64, buf2);
    expect(b1.overlaps(b2)).toBe(false);
  });

  it('allocations are aligned to pool alignment', () => {
    const pool = new MemoryPool('global', 128);
    const buf = new Buffer('a', [10], 'f32', 'global');
    const block = pool.allocate(100, buf);
    expect(block.offset % 128).toBe(0);
    expect(block.size % 128).toBe(0);
  });

  it('peakUsage tracks maximum memory usage', () => {
    const pool = new MemoryPool('global', 64);
    const buf1 = new Buffer('a', [16], 'f32', 'global');
    const buf2 = new Buffer('b', [16], 'f32', 'global');
    pool.allocate(256, buf1);
    pool.allocate(128, buf2);
    expect(pool.peakUsage).toBeGreaterThanOrEqual(256 + 128);
  });

  it('release frees a block allowing reuse', () => {
    const pool = new MemoryPool('global', 64);
    const buf1 = new Buffer('a', [16], 'f32', 'global');
    const buf2 = new Buffer('b', [16], 'f32', 'global');
    const b1 = pool.allocate(64, buf1);
    pool.release(b1);
    const b2 = pool.allocate(64, buf2);
    expect(b2.offset).toBe(0);
  });

  it('release of non-existent block is a no-op', () => {
    const pool = new MemoryPool('global', 64);
    const fakeBlock = new MemoryBlock(0, 64, null);
    pool.release(fakeBlock);
    expect(pool.blocks.length).toBe(0);
  });

  function buildGappedPool(strategy) {
    const pool = new MemoryPool('global', 64, strategy);
    const buf = () => new Buffer('x', [1], 'f32', 'global');
    pool.allocate(64, buf());
    const big = pool.allocate(192, buf());
    pool.allocate(64, buf());
    const mid = pool.allocate(128, buf());
    pool.allocate(64, buf());
    pool.release(big);
    pool.release(mid);
    return pool;
  }

  it('best-fit picks the tightest gap, first-fit the lowest', () => {
    const firstFit = buildGappedPool('first-fit');
    const bestFit = buildGappedPool('best-fit');

    const fa = firstFit.allocate(128, new Buffer('f', [1], 'f32', 'global'));
    const ba = bestFit.allocate(128, new Buffer('b', [1], 'f32', 'global'));

    expect(fa.offset).toBe(64);
    expect(ba.offset).toBe(320);
  });

  it('fragmentation reports the fraction of peak not occupied by live blocks', () => {
    const pool = buildGappedPool('best-fit');
    expect(pool.fragmentation()).toBeGreaterThan(0);
    expect(pool.fragmentation()).toBeLessThanOrEqual(1);
  });
});

describe('BufferAssignment', () => {
  it('assigns non-overlapping offsets to interfering intervals', () => {
    const bufA = new Buffer('a', [16], 'f32', 'global');
    const bufB = new Buffer('b', [16], 'f32', 'global');
    const intervalA = new BufferInterval(bufA, 0, 3, 'global');
    const intervalB = new BufferInterval(bufB, 1, 4, 'global');

    const assignment = new BufferAssignment();
    assignment.assign([intervalA, intervalB]);

    const offA = assignment.getOffset(bufA);
    const offB = assignment.getOffset(bufB);
    expect(offA).toBeGreaterThanOrEqual(0);
    expect(offB).toBeGreaterThanOrEqual(0);
    expect(offA).not.toBe(offB);
  });

  it('reuses memory for non-overlapping intervals', () => {
    const bufA = new Buffer('a', [16], 'f32', 'global');
    const bufB = new Buffer('b', [16], 'f32', 'global');
    const intervalA = new BufferInterval(bufA, 0, 1, 'global');
    const intervalB = new BufferInterval(bufB, 3, 5, 'global');

    const assignment = new BufferAssignment();
    assignment.assign([intervalA, intervalB]);

    const offA = assignment.getOffset(bufA);
    const offB = assignment.getOffset(bufB);
    expect(offA).toBe(offB);
  });

  it('sorts by size descending — larger buffers assigned first', () => {
    const big = new Buffer('big', [64], 'f32', 'global');
    const small = new Buffer('small', [4], 'f32', 'global');
    const intervalBig = new BufferInterval(big, 0, 2, 'global');
    const intervalSmall = new BufferInterval(small, 0, 2, 'global');

    const assignment = new BufferAssignment();
    assignment.assign([intervalSmall, intervalBig]);

    expect(assignment.getOffset(big)).toBe(0);
  });

  it('peakMemory returns total peak across all pools', () => {
    const buf = new Buffer('a', [64], 'f32', 'global');
    const interval = new BufferInterval(buf, 0, 1, 'global');

    const assignment = new BufferAssignment();
    assignment.assign([interval]);

    expect(assignment.peakMemory()).toBeGreaterThan(0);
  });

  it('peakMemory with scope returns peak for that scope only', () => {
    const buf = new Buffer('a', [16], 'f32', 'global');
    const interval = new BufferInterval(buf, 0, 1, 'global');

    const assignment = new BufferAssignment();
    assignment.assign([interval]);

    expect(assignment.peakMemory('global')).toBeGreaterThan(0);
    expect(assignment.peakMemory('shared')).toBe(0);
  });

  function noOverlapViolations(intervals, strategy) {
    const a = new BufferAssignment();
    a.assign(intervals, [], 64, strategy);
    let violations = 0;
    for (let i = 0; i < intervals.length; i++) {
      for (let j = i + 1; j < intervals.length; j++) {
        if (!intervals[i].overlaps(intervals[j])) continue;
        const x = a.getAssignment(intervals[i].buffer);
        const y = a.getAssignment(intervals[j].buffer);
        if (x.offset < y.offset + y.size && y.offset < x.offset + x.size) violations++;
      }
    }
    return violations;
  }

  it('simultaneously-live buffers never share memory (size-desc release-order regression)', () => {
    const iv = (name, sizeWords, a, b) => new BufferInterval(new Buffer(name, [sizeWords], 'f32', 'global'), a, b, 'global');
    const intervals = [iv('b0', 48, 0, 2), iv('b1', 48, 0, 0), iv('b2', 48, 3, 5), iv('b3', 16, 0, 3)];

    for (const strategy of ['best-fit', 'interference']) {
      expect(noOverlapViolations(intervals, strategy), strategy).toBe(0);
    }
  });

  it('interference allocation is conflict-free across many random interval sets', () => {
    let seed = 12345;
    const rng = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    for (let t = 0; t < 200; t++) {
      const k = 3 + Math.floor(rng() * 5);
      const intervals = [];
      for (let i = 0; i < k; i++) {
        const a = Math.floor(rng() * 6);
        const b = a + Math.floor(rng() * 4);
        const sizeWords = 16 * (1 + Math.floor(rng() * 3));
        intervals.push(new BufferInterval(new Buffer('b' + i, [sizeWords], 'f32', 'global'), a, b, 'global'));
      }
      expect(noOverlapViolations(intervals, 'best-fit'), `seed ${t} best-fit`).toBe(0);
      expect(noOverlapViolations(intervals, 'interference'), `seed ${t} interference`).toBe(0);
    }
  });

  it('inplace candidates share the same offset', () => {
    const srcBuf = new Buffer('src', [16], 'f32', 'global');
    const dstBuf = new Buffer('dst', [16], 'f32', 'global');
    const srcInterval = new BufferInterval(srcBuf, 0, 1, 'global');
    const dstInterval = new BufferInterval(dstBuf, 2, 3, 'global');

    const assignment = new BufferAssignment();
    assignment.assign(
      [srcInterval, dstInterval],
      [{ srcBuffer: srcBuf, dstBuffer: dstBuf }]
    );

    expect(assignment.getOffset(srcBuf)).toBe(assignment.getOffset(dstBuf));
    const dstAssignment = assignment.getAssignment(dstBuf);
    expect(dstAssignment.inplaceOf).toBe(srcBuf);
  });

  it('getAssignment returns null for unknown buffer', () => {
    const assignment = new BufferAssignment();
    const unknown = new Buffer('x', [4], 'f32', 'global');
    expect(assignment.getAssignment(unknown)).toBeNull();
  });

  it('getOffset returns -1 for unknown buffer', () => {
    const assignment = new BufferAssignment();
    const unknown = new Buffer('x', [4], 'f32', 'global');
    expect(assignment.getOffset(unknown)).toBe(-1);
  });

  it('zero-size intervals are skipped', () => {
    const buf = new Buffer('empty', [0], 'f32', 'global');
    const interval = new BufferInterval(buf, 0, 1, 'global');

    const assignment = new BufferAssignment();
    assignment.assign([interval]);

    expect(assignment.getAssignment(buf)).toBeNull();
  });

  it('multiple scopes get separate pools', () => {
    const globalBuf = new Buffer('g', [16], 'f32', 'global');
    const sharedBuf = new Buffer('s', [16], 'f32', 'shared');
    const intervalG = new BufferInterval(globalBuf, 0, 1, 'global');
    const intervalS = new BufferInterval(sharedBuf, 0, 1, 'shared');

    const assignment = new BufferAssignment();
    assignment.assign([intervalG, intervalS]);

    expect(assignment.pools.size).toBe(2);
    expect(assignment.pools.has('global')).toBe(true);
    expect(assignment.pools.has('shared')).toBe(true);
  });

  it('releases correct active entries when multiple expire at once (descending splice)', () => {
    const size = [16];
    const spec = [[0, 3, 9], [1, 5, 7], [2, 1, 1], [3, 4, 8], [4, 5, 5], [5, 2, 4], [6, 0, 1], [7, 0, 2]];
    const intervals = spec.map(([k, f, l]) =>
      new BufferInterval(new Buffer('b' + k, size, 'f32', 'global'), f, l, 'global'));

    const assignment = new BufferAssignment();
    assignment.assign(intervals);

    const bufSize = 16 * 4;
    const aligned = Math.ceil(bufSize / 64) * 64;
    expect(assignment.peakMemory()).toBe(aligned * 4);
  });

  it('many non-overlapping intervals reuse memory efficiently', () => {
    const bufs = [];
    const intervals = [];
    for (let i = 0; i < 10; i++) {
      const buf = new Buffer(`b${i}`, [16], 'f32', 'global');
      bufs.push(buf);
      intervals.push(new BufferInterval(buf, i * 2, i * 2 + 1, 'global'));
    }

    const assignment = new BufferAssignment();
    assignment.assign(intervals);

    const singleBufSize = 16 * 4;
    const aligned = Math.ceil(singleBufSize / 64) * 64;
    expect(assignment.peakMemory()).toBe(aligned);
  });
});

import { InplaceCandidate } from '../../../src/compiler/passes/memory/inplace_analysis.js';

describe('BufferAssignment: inplace destination extends aliased storage lifetime (no-alias-while-live)', () => {
  it('a later buffer does not reuse storage still held by a live inplace destination', () => {
    const src = new Buffer('src', [16], 'f32', 'global');
    const dst = new Buffer('dst', [16], 'f32', 'global');
    const other = new Buffer('other', [16], 'f32', 'global');
    // src dies at 1; dst aliases src in-place and lives [1,5]; other lives [2,5] (overlaps dst).
    const intervals = [
      new BufferInterval(src, 0, 1, 'global'),
      new BufferInterval(dst, 1, 5, 'global'),
      new BufferInterval(other, 2, 5, 'global'),
    ];
    const asg = new BufferAssignment();
    asg.assign(intervals, [new InplaceCandidate(src, dst, 'test')], 64);

    const oSrc = asg.getOffset(src);
    const oDst = asg.getOffset(dst);
    const oOther = asg.getOffset(other);

    // dst aliases src (in-place) — same storage by design.
    expect(oDst).toBe(oSrc);
    // other overlaps dst's live range [2,5] vs [1,5] → must NOT share storage with the live inplace dst.
    expect(oOther).not.toBe(oDst);
  });

  it('inplace chain (a→b→c) keeps the shared storage live until the last destination dies', () => {
    const a = new Buffer('a', [16], 'f32', 'global');
    const b = new Buffer('b', [16], 'f32', 'global');
    const c = new Buffer('c', [16], 'f32', 'global');
    const other = new Buffer('other', [16], 'f32', 'global');
    const intervals = [
      new BufferInterval(a, 0, 1, 'global'),
      new BufferInterval(b, 1, 2, 'global'),
      new BufferInterval(c, 2, 6, 'global'),
      new BufferInterval(other, 3, 6, 'global'),
    ];
    const asg = new BufferAssignment();
    asg.assign(intervals, [new InplaceCandidate(a, b, 't'), new InplaceCandidate(b, c, 't')], 64);
    expect(asg.getOffset(b)).toBe(asg.getOffset(a));
    expect(asg.getOffset(c)).toBe(asg.getOffset(a));
    // other overlaps c [3,6] which still holds a's storage → must not collide.
    expect(asg.getOffset(other)).not.toBe(asg.getOffset(a));
  });
});
