import { describe, it, expect } from 'vitest';
import { MemoryPool, MemoryBlock, BufferAssignment } from '../../../../src/compiler/passes/memory/buffer_assignment.js';
import { BufferInterval } from '../../../../src/compiler/passes/memory/buffer_liveness.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { InplaceCandidate } from '../../../../src/compiler/passes/memory/inplace_analysis.js';

describe('MemoryBlock', () => {
  it('end equals offset + size', () => {
    const block = new MemoryBlock(64, 128, null);
    expect(block.end).toBe(192);
  });
});

describe('MemoryPool', () => {
  it('placeAt aligns size up and tracks peak usage', () => {
    const pool = new MemoryPool('global', 128);
    const block = pool.placeAt(0, 100, new Buffer('a', [25], 'f32', 'global'));
    expect(block.size % 128).toBe(0);
    expect(pool.peakUsage).toBe(block.end);
  });
});

describe('BufferAssignment gap strategy', () => {
  // A, C, E stay live across the probe; B and D die before it, so the probe sees
  // two reusable holes: a 192-byte one at 256 and a 64-byte one at 576.
  function assignWithStrategy(strategy) {
    const mk = (name, numel) => new Buffer(name, [numel], 'f32', 'global');
    const probe = mk('probe', 16);
    const intervals = [
      new BufferInterval(mk('a', 64), 0, 20, 'global'),
      new BufferInterval(mk('b', 48), 0, 5, 'global'),
      new BufferInterval(mk('c', 32), 0, 20, 'global'),
      new BufferInterval(mk('d', 16), 0, 6, 'global'),
      new BufferInterval(mk('e', 16), 0, 20, 'global'),
      new BufferInterval(probe, 10, 11, 'global'),
    ];
    const assignment = new BufferAssignment().assign(intervals, [], 64, strategy);
    return assignment.getOffset(probe);
  }

  it('first-fit takes the lowest hole, best-fit the tightest', () => {
    const firstFit = assignWithStrategy('first-fit');
    const bestFit = assignWithStrategy('best-fit');
    expect(firstFit).toBe(256);
    expect(bestFit).toBe(576);
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

describe('BufferAssignment: inplace destination extends aliased storage lifetime (no-alias-while-live)', () => {
  it('a later buffer does not reuse storage still held by a live inplace destination', () => {
    const src = new Buffer('src', [16], 'f32', 'global');
    const dst = new Buffer('dst', [16], 'f32', 'global');
    const other = new Buffer('other', [16], 'f32', 'global');
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

    expect(oDst, 'dst aliases src in-place and must share its storage').toBe(oSrc);
    expect(oOther, 'other lives [2,5] overlapping the live inplace dst [1,5] and must not share its storage').not.toBe(oDst);
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
    expect(asg.getOffset(other), 'other lives [3,6] overlapping c which still holds a\'s storage and must not collide').not.toBe(asg.getOffset(a));
  });
});
