import { describe, it, expect } from 'vitest';
import { flattenRowMajorIndex } from '../../src/backend/index_emit.js';

const leaf = (e) => e;
const dyn = (b, i) => `_st(${i})`;

describe('flattenRowMajorIndex (shared backend index emitter)', () => {
  it('returns "0" for a scalar (no indices)', () => {
    expect(flattenRowMajorIndex({ strides: [] }, [], leaf, dyn)).toBe('0');
  });

  it('returns the bare leaf for a single index (no stride math)', () => {
    expect(flattenRowMajorIndex({ strides: [1] }, ['i'], leaf, dyn)).toBe('i');
  });

  it('multiplies by numeric strides and joins (stride 1 elided)', () => {
    expect(flattenRowMajorIndex({ strides: [4, 1] }, ['i', 'j'], leaf, dyn, false)).toBe('i * 4 + j');
  });

  it('skipZero drops 0-valued index terms (CPU mode)', () => {
    expect(flattenRowMajorIndex({ strides: [4, 1] }, ['0', 'j'], leaf, dyn, true)).toBe('j');
    expect(flattenRowMajorIndex({ strides: [4, 1] }, ['0', '0'], leaf, dyn, true)).toBe('0');
  });

  it('without skipZero keeps every term (CUDA/WebGPU mode)', () => {
    expect(flattenRowMajorIndex({ strides: [4, 1] }, ['0', 'j'], leaf, dyn, false)).toBe('0 * 4 + j');
  });

  it('falls back to the dynamic-stride callback for non-numeric strides', () => {
    expect(flattenRowMajorIndex({ strides: [-1, 1] }, ['i', 'j'], leaf, dyn, false)).toBe('i * _st(0) + j');
  });
});
