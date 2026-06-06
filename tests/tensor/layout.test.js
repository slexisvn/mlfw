import { describe, it, expect } from 'vitest';
import {
  MemoryFormat,
  memoryFormatPermutation,
  isChannelsLast,
} from '../../src/tensor/types/layout.js';

describe('memoryFormatPermutation', () => {
  it('returns [0,2,3,1] for 4D channels_last', () => {
    expect(memoryFormatPermutation(4, MemoryFormat.CHANNELS_LAST)).toEqual([0, 2, 3, 1]);
  });

  it('returns [0,2,3,4,1] for 5D channels_last', () => {
    expect(memoryFormatPermutation(5, MemoryFormat.CHANNELS_LAST)).toEqual([0, 2, 3, 4, 1]);
  });

  it('returns null for contiguous format', () => {
    expect(memoryFormatPermutation(4, MemoryFormat.CONTIGUOUS)).toBeNull();
  });

  it('returns null for preserve format', () => {
    expect(memoryFormatPermutation(4, MemoryFormat.PRESERVE)).toBeNull();
  });

  it('returns null for channels_last with unsupported ndim', () => {
    expect(memoryFormatPermutation(3, MemoryFormat.CHANNELS_LAST)).toBeNull();
    expect(memoryFormatPermutation(6, MemoryFormat.CHANNELS_LAST)).toBeNull();
  });
});

describe('isChannelsLast', () => {
  it('detects NCHW→NHWC layout for 4D tensor', () => {
    const sizes = [1, 3, 4, 4];
    const strides = [48, 1, 12, 3];
    expect(isChannelsLast(sizes, strides)).toBe(true);
  });

  it('rejects standard contiguous 4D tensor (NCHW)', () => {
    const sizes = [1, 3, 4, 4];
    const strides = [48, 16, 4, 1];
    expect(isChannelsLast(sizes, strides)).toBe(false);
  });

  it('returns false for tensors with fewer than 4 dims', () => {
    expect(isChannelsLast([3, 4, 5], [20, 5, 1])).toBe(false);
  });

  it('returns false when any size is 0', () => {
    const sizes = [1, 0, 4, 4];
    const strides = [0, 1, 0, 0];
    expect(isChannelsLast(sizes, strides)).toBe(false);
  });

  it('detects channels_last for 5D tensor (NCDHW→NDHWC)', () => {
    const sizes = [1, 3, 2, 4, 4];
    const strides = [96, 1, 48, 12, 3];
    expect(isChannelsLast(sizes, strides)).toBe(true);
  });
});
