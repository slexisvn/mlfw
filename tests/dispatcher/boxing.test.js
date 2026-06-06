import { describe, it, expect } from 'vitest';
import { KernelFunction, IValue, IValueTag } from '../../src/dispatcher/boxing.js';
import { DispatchKeySet, DispatchKey, EMPTY_KEY_SET } from '../../src/dispatcher/dispatch_key.js';
import { tensor } from '../../src/index.js';

describe('KernelFunction.fromUnboxed', () => {
  it('callUnboxed passes keySet and args to the function', () => {
    let receivedKs, receivedArgs;
    const kf = KernelFunction.fromUnboxed((ks, a, b) => {
      receivedKs = ks;
      receivedArgs = [a, b];
      return a;
    });

    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    const t = tensor([1, 2]);
    kf.callUnboxed(ks, t, 42);

    expect(receivedKs).toBe(ks);
    expect(receivedArgs[0]).toBe(t);
    expect(receivedArgs[1]).toBe(42);
  });

  it('callUnboxed returns the kernel result unchanged', () => {
    const t = tensor([10, 20]);
    const kf = KernelFunction.fromUnboxed((ks, self) => self);
    const result = kf.callUnboxed(EMPTY_KEY_SET, t);
    expect(result).toBe(t);
  });
});

describe('KernelFunction.fromBoxed', () => {
  it('callBoxed passes IValue stack to the function', () => {
    let receivedStack;
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      receivedStack = stack;
      return [stack[0]];
    });

    const iv = IValue.tensor(tensor([5]));
    kf.callBoxed(EMPTY_KEY_SET, [iv]);
    expect(receivedStack[0]).toBe(iv);
  });
});

describe('boxed-unboxed conversion', () => {
  it('fromBoxed called via callUnboxed auto-wraps args to IValues', () => {
    let receivedStack;
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      receivedStack = stack;
      return [stack[0]];
    });

    const t = tensor([1, 2, 3]);
    kf.callUnboxed(EMPTY_KEY_SET, t, 42);

    expect(receivedStack[0].isTensor()).toBe(true);
    expect(receivedStack[0].toTensor()).toBe(t);
    expect(receivedStack[1].isInt()).toBe(true);
    expect(receivedStack[1].toInt()).toBe(42);
  });

  it('fromUnboxed called via callBoxed unwraps IValues to raw args', () => {
    let receivedArgs;
    const kf = KernelFunction.fromUnboxed((ks, a, b) => {
      receivedArgs = [a, b];
      return a;
    });

    const t = tensor([7]);
    const stack = [IValue.tensor(t), IValue.int(3)];
    kf.callBoxed(EMPTY_KEY_SET, stack);

    expect(receivedArgs[0]).toBe(t);
    expect(receivedArgs[1]).toBe(3);
  });

  it('fromBoxed via callUnboxed unwraps single-element result', () => {
    const t = tensor([99]);
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      return [IValue.tensor(t)];
    });

    const result = kf.callUnboxed(EMPTY_KEY_SET, tensor([1]));
    expect(result).toBe(t);
  });
});

describe('IValue auto-wrapping', () => {
  it('wraps tensor object as TENSOR', () => {
    const t = tensor([1]);
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.TENSOR);
      return [stack[0]];
    });
    kf.callUnboxed(EMPTY_KEY_SET, t);
  });

  it('wraps integer as INT', () => {
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.INT);
      expect(stack[0].toInt()).toBe(7);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, 7);
  });

  it('wraps float as FLOAT', () => {
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.FLOAT);
      expect(stack[0].toFloat()).toBeCloseTo(3.14);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, 3.14);
  });

  it('wraps boolean as BOOL', () => {
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.BOOL);
      expect(stack[0].toBool()).toBe(true);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, true);
  });

  it('wraps null as NONE', () => {
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.NONE);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, null);
  });

  it('wraps array of tensors as TENSOR_LIST', () => {
    const t1 = tensor([1]);
    const t2 = tensor([2]);
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.TENSOR_LIST);
      const list = stack[0].toTensorList();
      expect(list.length).toBe(2);
      expect(list[0]).toBe(t1);
      expect(list[1]).toBe(t2);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, [t1, t2]);
  });

  it('wraps array of ints as INT_LIST', () => {
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.INT_LIST);
      expect(stack[0].toIntList()).toEqual([1, 2, 3]);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, [1, 2, 3]);
  });

  it('wraps string as STRING', () => {
    const kf = KernelFunction.fromBoxed((ks, stack) => {
      expect(stack[0].tag).toBe(IValueTag.STRING);
      return [];
    });
    kf.callUnboxed(EMPTY_KEY_SET, 'hello');
  });
});

describe('KernelFunction.fromBoth', () => {
  it('callUnboxed uses unboxed path directly', () => {
    let path;
    const kf = KernelFunction.fromBoth(
      (ks, stack) => { path = 'boxed'; return []; },
      (ks, x) => { path = 'unboxed'; return x; }
    );
    kf.callUnboxed(EMPTY_KEY_SET, tensor([1]));
    expect(path).toBe('unboxed');
  });

  it('callBoxed uses boxed path directly', () => {
    let path;
    const kf = KernelFunction.fromBoth(
      (ks, stack) => { path = 'boxed'; return []; },
      (ks, x) => { path = 'unboxed'; return x; }
    );
    kf.callBoxed(EMPTY_KEY_SET, [IValue.tensor(tensor([1]))]);
    expect(path).toBe('boxed');
  });
});
