import { describe, it, expect } from 'vitest';
import { DispatchKey, DispatchKeySet } from '../../src/dispatcher/dispatch_key.js';
import { KernelFunction } from '../../src/dispatcher/boxing.js';
import { parseSchema } from '../../src/dispatcher/operator_schema.js';
import { dispatcher, computeKeySet } from '../../src/dispatcher/dispatcher.js';
import { tensor } from '../../src/index.js';
import { jitCompile, jitCacheClear } from '../../src/dispatcher/jit_cache.js';
import { CPUTarget, WasmTarget } from '../../src/backend/target.js';
import { ScalarType } from '../../src/compiler/ir/graph/types.js';

describe('dispatch routes to correct kernel by key priority', () => {
  it('redispatch hits highest priority key first, then next', () => {
    const schema = parseSchema('_test_priority(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);

    const order = [];
    const autogradKernel = KernelFunction.fromUnboxed((ks, self) => {
      order.push('autograd');
      return dispatcher.redispatch(handle, ks, self);
    });
    const cpuKernel = KernelFunction.fromUnboxed((ks, self) => {
      order.push('cpu');
      return self;
    });

    handle.entry.registerKernel(DispatchKey.CPU, cpuKernel);
    handle.entry.registerKernel(DispatchKey.AUTOGRAD, autogradKernel);

    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    dispatcher.redispatch(handle, ks, tensor([1]));

    expect(order).toEqual(['autograd', 'cpu']);
  });

  it('throws when no kernel is registered for any key', () => {
    const schema = parseSchema('_test_no_kernel(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);
    const ks = DispatchKeySet.fromKey(DispatchKey.WASM);

    expect(() => dispatcher.dispatch(handle, ks, tensor([1]))).toThrow(/No kernel/);
  });
});

describe('catchAll kernel', () => {
  it('used as last resort when no key-specific kernel matches', () => {
    const schema = parseSchema('_test_catchall(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);

    let caught = false;
    handle.entry.setCatchAll(KernelFunction.fromUnboxed((ks, self) => {
      caught = true;
      return self;
    }));

    const ks = DispatchKeySet.fromKey(DispatchKey.META);
    dispatcher.dispatch(handle, ks, tensor([1]));
    expect(caught).toBe(true);
  });
});

describe('computeKeySet', () => {
  it('unions dispatch key sets from tensor arguments', () => {
    const a = tensor([1]);
    const b = tensor([2]);
    const ks = computeKeySet([a, b], null);
    expect(ks.has(DispatchKey.CPU)).toBe(true);
  });

  it('skips non-tensor arguments', () => {
    const a = tensor([1]);
    const ks = computeKeySet([a, 42, 'hello'], null);
    expect(ks.has(DispatchKey.CPU)).toBe(true);
    expect(ks.count()).toBeGreaterThanOrEqual(1);
  });

  it('uses schema tensorArgIndices when provided', () => {
    const schema = parseSchema('_test_keyset(Tensor a, int dim) -> Tensor', 'mlc');
    const a = tensor([1]);
    const ks = computeKeySet([a, 0], schema);
    expect(ks.has(DispatchKey.CPU)).toBe(true);
  });
});

describe('findOp', () => {
  it('returns null for unregistered op', () => {
    expect(dispatcher.findOp('_nonexistent_op_xyz')).toBeNull();
  });
});

describe('three-level dispatch chain', () => {
  it('dispatches through tracing → autograd → cpu in order', () => {
    const schema = parseSchema('_test_3level(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);

    const order = [];
    handle.entry.registerKernel(DispatchKey.TRACING, KernelFunction.fromUnboxed((ks, self) => {
      order.push('tracing');
      return dispatcher.redispatch(handle, ks, self);
    }));
    handle.entry.registerKernel(DispatchKey.AUTOGRAD, KernelFunction.fromUnboxed((ks, self) => {
      order.push('autograd');
      return dispatcher.redispatch(handle, ks, self);
    }));
    handle.entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed((ks, self) => {
      order.push('cpu');
      return self;
    }));

    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD, DispatchKey.TRACING);
    dispatcher.redispatch(handle, ks, tensor([1]));
    expect(order).toEqual(['tracing', 'autograd', 'cpu']);
  });
});

describe('kernel replacement', () => {
  it('new kernel overwrites previous for same dispatch key', () => {
    const schema = parseSchema('_test_overwrite(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);

    let version = 0;
    handle.entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed((ks, self) => {
      version = 1;
      return self;
    }));
    handle.entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed((ks, self) => {
      version = 2;
      return self;
    }));

    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    dispatcher.redispatch(handle, ks, tensor([1]));
    expect(version).toBe(2);
  });
});

describe('kernel transforms result', () => {
  it('autograd wrapper can modify result before returning', () => {
    const schema = parseSchema('_test_transform(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);

    handle.entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed((ks, self) => {
      return self;
    }));
    handle.entry.registerKernel(DispatchKey.AUTOGRAD, KernelFunction.fromUnboxed((ks, self) => {
      const result = dispatcher.redispatch(handle, ks, self);
      result._testMarker = 'wrapped';
      return result;
    }));

    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const out = dispatcher.redispatch(handle, ks, tensor([1]));
    expect(out._testMarker).toBe('wrapped');
  });
});

describe('registerOp deduplication', () => {
  it('registering same schema twice returns same handle', () => {
    const s1 = parseSchema('_test_dedup(Tensor self) -> Tensor', 'mlc');
    const s2 = parseSchema('_test_dedup(Tensor self) -> Tensor', 'mlc');
    const h1 = dispatcher.registerOp(s1);
    const h2 = dispatcher.registerOp(s2);
    expect(h1).toBe(h2);
  });
});

describe('dispatch with empty keyset after stripping', () => {
  it('throws when all keys consumed by redispatch chain', () => {
    const schema = parseSchema('_test_exhaust(Tensor self) -> Tensor', 'mlc');
    const handle = dispatcher.registerOp(schema);

    handle.entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed((ks, self) => {
      return dispatcher.redispatch(handle, ks, self);
    }));

    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    expect(() => dispatcher.redispatch(handle, ks, tensor([1]))).toThrow();
  });
});

describe('jit cache key distinguishes op / shape / dtype / target', () => {
  const F = ScalarType.F32, I = ScalarType.I32;
  const ta = (shape, dtype) => ({ shape, dtype });

  it('same op+shape+dtype+target returns the identical cached entry', () => {
    jitCacheClear();
    const a = jitCompile('relu', [ta([4, 4], F)], {}, CPUTarget());
    const b = jitCompile('relu', [ta([4, 4], F)], {}, CPUTarget());
    expect(a).toBe(b);
  });

  it('differing dtype yields distinct entries (i32 reduce-init must not leak through cache)', () => {
    jitCacheClear();
    const f = jitCompile('sum', [ta([4, 4], F)], {}, CPUTarget());
    const i = jitCompile('sum', [ta([4, 4], I)], {}, CPUTarget());
    expect(f).not.toBe(i);
    expect(f.outDtype).toBe('f32');
    expect(i.outDtype).toBe('i32');
  });

  it('differing shape yields distinct entries', () => {
    jitCacheClear();
    const a = jitCompile('relu', [ta([4, 4], F)], {}, CPUTarget());
    const b = jitCompile('relu', [ta([8, 4], F)], {}, CPUTarget());
    expect(a).not.toBe(b);
  });

  it('differing target kind yields distinct entries on distinct runtimes', () => {
    jitCacheClear();
    const cpu = jitCompile('relu', [ta([4, 4], F)], {}, CPUTarget());
    const wasm = jitCompile('relu', [ta([4, 4], F)], {}, WasmTarget());
    expect(cpu).not.toBe(wasm);
    expect(cpu.runtime).not.toBe(wasm.runtime);
  });

  it('differing scalar args (reduction dim) yields distinct entries', () => {
    jitCacheClear();
    const a = jitCompile('sum', [ta([4, 6], F)], { dim: 0 }, CPUTarget());
    const b = jitCompile('sum', [ta([4, 6], F)], { dim: 1 }, CPUTarget());
    expect(a).not.toBe(b);
  });
});

