import { describe, it, expect } from 'vitest';
import { RuntimeMemoryManager, RuntimeModule } from '../../../src/compiler/runtime/runtime.js';

describe('RuntimeMemoryManager pool reuse', () => {
  it('does not discard a pooled buffer that is too small for the request', () => {
    const mm = new RuntimeMemoryManager();
    const small = mm.allocate(16);
    mm.release(small, 16);
    const big = mm.allocate(256);
    expect(big.length).toBeGreaterThanOrEqual(64);
    mm.release(big, 256);
    const reuseSmall = mm.allocate(16);
    expect(reuseSmall).toBe(small);
  });

  it('reuses the smallest fitting buffer from the pool', () => {
    const mm = new RuntimeMemoryManager();
    const a = mm.allocate(64);
    const b = mm.allocate(256);
    mm.release(a, 64);
    mm.release(b, 256);
    const req = mm.allocate(64);
    expect(req).toBe(a);
  });
});

describe('RuntimeModule._extractShapeParams tensor identity', () => {
  it('resolves each shape param from its own named buffer', () => {
    const shapeParamMap = new Map([
      ['A:0', { name: 'm' }],
      ['B:0', { name: 'n' }],
    ]);
    const bufferMap = new Map([
      ['A', {}],
      ['B', {}],
    ]);
    const tensorShapes = new Map([
      [0, [7]],
      [1, [11]],
    ]);
    const result = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, [], bufferMap);
    expect(result).toEqual([7, 11]);
  });

  it('falls back to first matching tensor when buffer map is absent', () => {
    const shapeParamMap = new Map([['A:1', { name: 'm' }]]);
    const tensorShapes = new Map([[0, [3, 9]]]);
    const result = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, [], null);
    expect(result).toEqual([9]);
  });
});
