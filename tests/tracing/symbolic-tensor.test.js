import { describe, it, expect } from 'vitest';
import { tensor, trace } from '../../src/index.js';
import { DispatchKey } from '../../src/dispatcher/dispatch_key.js';
import { Tensor } from '../../src/tensor/core/tensor.js';

describe('SymbolicTensor properties during trace', () => {
  it('isSymbolic is true and dispatchKeySet includes TRACING', () => {
    const x = tensor([1, 2, 3]);
    let captured = null;
    trace((t) => {
      captured = t;
      return t.neg();
    }, [x]);

    expect(captured.isSymbolic).toBe(true);
    expect(captured.dispatchKeySet.has(DispatchKey.TRACING)).toBe(true);
  });

  it('symbolic tensor is instanceof Tensor', () => {
    const x = tensor([1, 2]);
    let captured = null;
    trace((t) => {
      captured = t;
      return t.neg();
    }, [x]);

    expect(captured instanceof Tensor).toBe(true);
  });

  it('shape and dtype match the original input', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    let captured = null;
    trace((t) => {
      captured = t;
      return t.neg();
    }, [x]);

    expect(captured.shape).toEqual([2, 3]);
    expect(captured.dtype).toBe(x.dtype);
  });
});

describe('tracer activate/deactivate isolation', () => {
  it('regular tensor has no TRACING key outside trace context', () => {
    const x = tensor([1, 2, 3]);
    expect(x.dispatchKeySet.has(DispatchKey.TRACING)).toBe(false);
    expect(x.isSymbolic).toBeUndefined();
  });
});
