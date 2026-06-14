import { describe, it, expect } from 'vitest';
import { RuntimeModule } from '../../../src/compiler/runtime/runtime.js';

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
