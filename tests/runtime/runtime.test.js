import { describe, it, expect } from 'vitest';
import { RuntimeModule } from '../../src/runtime/runtime.js';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/backend/target.js';

describe('RuntimeModule AOT serialize/deserialize (no recompile)', () => {
  it('reconstructs a runnable module from serialized kernels and runs identically', () => {
    const t = (s) => new TensorType(s, ScalarType.F32);
    const fn = buildFunction('addmul', [t([4]), t([4])], [t([4])], (b, [x, y]) => {
      b.returnOp([b.mul(b.add(x, y).getResult(0), x).getResult(0)]);
    });
    const rm = compileGraph(fn, CPUTarget()).module;
    const k = rm.listKernels()[0];
    const a = new Float32Array([1, 2, 3, 4]);
    const bb = new Float32Array([5, 6, 7, 8]);
    const o1 = new Float32Array(4);
    rm.run(k, a, bb, o1);

    const blob = JSON.parse(JSON.stringify(rm.serialize()));
    const restored = RuntimeModule.deserialize(blob);
    const o2 = new Float32Array(4);
    restored.run(restored.listKernels()[0], a, bb, o2);

    expect([...o2]).toEqual([...o1]);
    expect(JSON.stringify(restored.serialize())).toBe(JSON.stringify(blob));
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
