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
