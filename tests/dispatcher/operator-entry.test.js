import { describe, it, expect } from 'vitest';
import { OperatorEntry } from '../../src/dispatcher/operator_entry.js';
import { DispatchKey, DispatchKeySet } from '../../src/dispatcher/dispatch_key.js';
import { KernelFunction } from '../../src/dispatcher/boxing.js';
import { parseSchema } from '../../src/dispatcher/operator_schema.js';

function makeEntry() {
  return new OperatorEntry(parseSchema('_entry_test(Tensor x) -> Tensor'));
}

describe('bestKernel', () => {
  it('returns highest priority kernel from keyset', () => {
    const entry = makeEntry();
    const cpuK = KernelFunction.fromUnboxed((ks, x) => 'cpu');
    const autoK = KernelFunction.fromUnboxed((ks, x) => 'autograd');
    entry.registerKernel(DispatchKey.CPU, cpuK);
    entry.registerKernel(DispatchKey.AUTOGRAD, autoK);

    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const best = entry.bestKernel(ks);
    expect(best.key).toBe(DispatchKey.AUTOGRAD);
    expect(best.kernel).toBe(autoK);
  });

  it('returns catchAll when no key-specific kernel matches', () => {
    const entry = makeEntry();
    const catchK = KernelFunction.fromUnboxed((ks, x) => 'catch');
    entry.setCatchAll(catchK);

    const ks = DispatchKeySet.fromKey(DispatchKey.WASM);
    const best = entry.bestKernel(ks);
    expect(best.key).toBe(-1);
    expect(best.kernel).toBe(catchK);
  });

  it('returns null when no kernel and no catchAll', () => {
    const entry = makeEntry();
    const ks = DispatchKeySet.fromKey(DispatchKey.WASM);
    expect(entry.bestKernel(ks)).toBeNull();
  });

  it('prefers key-specific kernel over catchAll', () => {
    const entry = makeEntry();
    const cpuK = KernelFunction.fromUnboxed((ks, x) => 'cpu');
    const catchK = KernelFunction.fromUnboxed((ks, x) => 'catch');
    entry.registerKernel(DispatchKey.CPU, cpuK);
    entry.setCatchAll(catchK);

    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    const best = entry.bestKernel(ks);
    expect(best.kernel).toBe(cpuK);
  });
});

describe('removeKernel', () => {
  it('makes previously registered kernel unavailable', () => {
    const entry = makeEntry();
    const k = KernelFunction.fromUnboxed((ks, x) => x);
    entry.registerKernel(DispatchKey.CPU, k);
    expect(entry.lookupKernel(DispatchKey.CPU)).toBe(k);

    entry.removeKernel(DispatchKey.CPU);
    expect(entry.lookupKernel(DispatchKey.CPU)).toBeNull();
  });
});

describe('registeredKeys', () => {
  it('returns only keys that have kernels registered', () => {
    const entry = makeEntry();
    entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed(() => {}));
    entry.registerKernel(DispatchKey.AUTOGRAD, KernelFunction.fromUnboxed(() => {}));

    const keys = entry.registeredKeys();
    expect(keys).toContain(DispatchKey.CPU);
    expect(keys).toContain(DispatchKey.AUTOGRAD);
    expect(keys).not.toContain(DispatchKey.GPU);
    expect(keys.length).toBe(2);
  });

  it('reflects removal', () => {
    const entry = makeEntry();
    entry.registerKernel(DispatchKey.CPU, KernelFunction.fromUnboxed(() => {}));
    entry.registerKernel(DispatchKey.GPU, KernelFunction.fromUnboxed(() => {}));
    entry.removeKernel(DispatchKey.CPU);

    const keys = entry.registeredKeys();
    expect(keys).not.toContain(DispatchKey.CPU);
    expect(keys).toContain(DispatchKey.GPU);
  });
});
