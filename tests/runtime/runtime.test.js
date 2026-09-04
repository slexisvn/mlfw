import { describe, it, expect } from 'vitest';
import { RuntimeModule } from '../../src/runtime/runtime.js';
import { registerBackend } from '../../src/runtime/backend_registry.js';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/compiler/support/target.js';

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

describe('RuntimeModule lazy instantiation', () => {
  it('keeps compiled kernels as source artifacts until execution', () => {
    let instantiateCount = 0;
    registerBackend('lazy-test', {
      instantiate(kernel) {
        instantiateCount++;
        return { kernel };
      },
      runSync(inst, tensorArgs) {
        tensorArgs[1].set(tensorArgs[0]);
        return inst;
      },
      async runAsync(inst, tensorArgs) {
        tensorArgs[1].set(tensorArgs[0]);
        return inst;
      },
      isAsync(inst, kernel) { return !!(kernel && kernel.metadata.async); },
    });

    const mod = new RuntimeModule('lazy');
    mod.addCompiledKernel({
      name: 'copy',
      source: 'source',
      target: { name: 'lazy' },
      metadata: { kind: 'lazy-test' },
    });

    expect(instantiateCount).toBe(0);
    expect(mod.listKernels()).toEqual(['copy']);
    expect(mod.getKernelSource('copy')).toBe('source');
    expect(mod.serialize().kernels).toHaveLength(1);
    expect(instantiateCount).toBe(0);

    const input = new Float32Array([1, 2]);
    const output = new Float32Array(2);
    mod.run('copy', input, output);

    expect([...output]).toEqual([1, 2]);
    expect(instantiateCount).toBe(1);

    mod.run('copy', input, output);
    expect(instantiateCount).toBe(1);
  });

  it('answers async capability from kernel metadata without instantiating', () => {
    let instantiateCount = 0;
    registerBackend('lazy-async-test', {
      instantiate(kernel) {
        instantiateCount++;
        return { kernel };
      },
      runSync() {},
      async runAsync() {},
      isAsync(inst, kernel) { return !!(kernel && kernel.metadata.async); },
    });

    const mod = new RuntimeModule('lazy-async');
    mod.addCompiledKernel({
      name: 'asyncKernel',
      source: 'source',
      target: { name: 'lazy' },
      metadata: { kind: 'lazy-async-test', async: true },
    });

    expect(mod.isAsync('asyncKernel')).toBe(true);
    expect(instantiateCount).toBe(0);
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

  it('prefers a buffer that is an argument over the first one the variable names', () => {
    const shapeParamMap = new Map([
      ['tmp:0', { name: 'm' }],
      ['A:0', { name: 'm' }],
    ]);
    const bufferMap = new Map([['A', {}], ['out', {}]]);
    const tensorShapes = new Map([[0, [7, 4]], [1, [7, 2]]]);
    const result = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, [], bufferMap);
    expect(result).toEqual([7]);
  });

  it('refuses to guess an extent no argument carries a shape for', () => {
    const shapeParamMap = new Map([['A:0', { name: 'm' }]]);
    const bufferMap = new Map([['A', {}]]);
    expect(() => RuntimeModule._extractShapeParams(shapeParamMap, new Map(), [], bufferMap))
      .toThrow(/dynamic extent 'm' cannot be resolved/);
  });
});
