import { describe, it, expect } from 'vitest';
import { tensor, Module, trace, where } from '../../src/index.js';

class DataDependent extends Module {
  forward(t) {
    if (t.sum().item() > 0) return t.mul(10);
    return t.mul(-1);
  }
}

function traceError(fn, inputs, opts) {
  try {
    trace(fn, inputs, opts);
  } catch (e) {
    return e;
  }
  throw new Error('expected trace to fail');
}

describe('reading a value from a symbolic tensor', () => {
  it('branching on item() reports data-dependent control flow instead of a null dereference', () => {
    const err = traceError((t) => new DataDependent().forward(t), [tensor([[1, 2], [3, 4]])]);

    expect(err.message).not.toMatch(/Cannot read properties of null/);
    expect(err.message).toContain('item() is not available on a symbolic tensor');
    expect(err.message).toContain('tracing records operations instead of computing them');
    expect(err.message).toContain('data-dependent control flow');
    expect(err.message).toContain('scan (src/tracing/scan.ts)');
    expect(err.message).toContain('dynamic_shapes');
  });

  it('names the accessor attempted and the tensor it was attempted on', () => {
    const dataErr = traceError((t) => t.neg().data, [tensor([[1, 2, 3], [4, 5, 6]])]);
    const arrayErr = traceError((t) => t.neg().toArray(), [tensor([1, 2, 3])]);
    const storageErr = traceError((t) => t.neg().storage, [tensor([1, 2])]);

    expect(dataErr.message).toContain('.data is not available on a symbolic tensor');
    expect(dataErr.message).toContain('shape [2, 3], dtype f32');
    expect(arrayErr.message).toContain('toArray() is not available on a symbolic tensor');
    expect(arrayErr.message).toContain('shape [3], dtype f32');
    expect(storageErr.message).toContain('.storage is not available on a symbolic tensor');
    expect(storageErr.message).toContain('data-dependent control flow');
  });

  it('iterating a symbolic tensor fails instead of yielding views over empty storage', () => {
    const err = traceError((t) => {
      let acc = null;
      for (const row of t) acc = acc === null ? row : acc.add(row);
      return acc;
    }, [tensor([[1, 2], [3, 4]])]);

    expect(err.message).toContain('iterating a tensor is not available on a symbolic tensor');
  });

  it('describes a dynamic dimension by its symbol rather than as -1', () => {
    const err = traceError((t) => t.neg().data, [tensor([[1, 2, 3], [4, 5, 6]])], { dynamicShapes: [new Set([0])] });

    expect(err.message).toContain('shape [s0, 3], dtype f32');
  });

  it('the remedy the message suggests traces: the branch becomes where()', () => {
    const graph = trace((t) => {
      const positive = t.sum().gt(0);
      return where(positive, t.mul(10), t.mul(-1));
    }, [tensor([[1, 2], [3, 4]])]);

    const fn = graph.functions().next().value;
    const opNames = [...fn.ops()].map((o) => o.opName);
    expect(opNames).toContain('where');
    expect(opNames.filter((n) => n === 'mul').length).toBe(2);
  });
});
