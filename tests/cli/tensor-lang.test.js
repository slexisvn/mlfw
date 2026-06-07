import { describe, expect, it } from 'vitest';
import { TensorLangRuntime } from '../../src/cli/runtime.js';
import { formatValue } from '../../src/cli/format.js';

describe('Tensor Lang', () => {
  it('evaluates tensor expressions including matmul', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      x = tensor([[1, 2]])
      w = tensor([[3], [4]])
      x @ w
    `);
    expect(result.shape).toEqual([1, 1]);
    expect(result.item()).toBe(11);
  });

  it('promotes scalars in tensor operators and function calls', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute('tensor([1, 2]) * 2 + 1').toArray()).toEqual([3, 5]);
    expect(runtime.execute('mul(tensor([1, 2]), 3)').toArray()).toEqual([3, 6]);
  });

  it('formats scalar and CPU tensors for the CLI', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const scalar = runtime.execute('tensor(2)');
    const vector = runtime.execute('tensor([1, 2])');

    expect(formatValue(scalar)).toBe('Tensor(2, dtype=f32)');
    expect(formatValue(vector)).toBe('Tensor(shape=[2], dtype=f32)\n[1,2]');
  });

  it('defines and runs a custom model', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      model MLP(input, hidden, output):
        fc1 = Linear(input, hidden)
        fc2 = Linear(hidden, output)

        forward x:
          x = relu(fc1(x))
          return fc2(x)

      model = MLP(4, 3, 2)
      x = randn([5, 4])
      model(x)
    `);
    expect(result.shape).toEqual([5, 2]);
  });

  it('compiles a model and records trace events', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      model = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
      x = randn([5, 4])
      compile(model, input=x)
    `);
    expect(result.events.length).toBeGreaterThan(0);
    expect(result.result.listKernels().length).toBeGreaterThan(0);
  });

  it('compiles tensor operators inside custom forward', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      model Residual:
        forward x:
          return relu(x + x)
      net = Residual()
      x = randn([2, 4])
      compile(net, input=x)
    `);
    expect(result.result.listKernels().length).toBeGreaterThan(0);
  });

  it('compiles scalar tensor operators inside custom forward', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      model Scale:
        forward x:
          return mul(x, 2) + 1
      net = Scale()
      x = randn([2, 4])
      compile(net, input=x)
    `);
    expect(result.result.listKernels().length).toBeGreaterThan(0);
  });

  it('indexes and slices tensors', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute('tensor([[1, 2, 3], [4, 5, 6]])[1]').toArray()).toEqual([4, 5, 6]);
    expect(runtime.execute('tensor([[1, 2, 3], [4, 5, 6]])[:, 1]').toArray()).toEqual([2, 5]);
    expect(runtime.execute('tensor([0, 1, 2, 3, 4])[1:5:2]').toArray()).toEqual([1, 3]);
    expect(runtime.execute('tensor([1, 2, 3])[-1]').item()).toBe(3);
  });

  it('exposes view and like-operation builtins', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute('transpose(reshape(onesLike(tensor([1, 2, 3, 4])), [2, 2]), 0, 1)');
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([[1, 1], [1, 1]]);
  });

  it('passes named convolution options as an options object', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const conv = runtime.execute('Conv2d(3, 8, 3, padding=1, bias=false)');
    expect(conv.padding).toBe(1);
    expect(conv.bias).toBeNull();
  });

  it('reports runtime errors at the source expression', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(() => runtime.execute('x = tensor([1])\nmissing(x)'))
      .toThrow(/Unknown name 'missing' at 2:1/);
  });

  it('supports basic autograd builtins', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      x = tensor([2], grad=true)
      y = sum(x * x)
      backward(y)
      grad(x)
    `);
    expect(result.toArray()).toEqual([4]);
  });

  it('evaluates scalar logical operators', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute('true and false')).toBe(false);
    expect(runtime.execute('true and true')).toBe(true);
    expect(runtime.execute('true or false')).toBe(true);
    expect(runtime.execute('false or false')).toBe(false);
    expect(runtime.execute('not true')).toBe(false);
    expect(runtime.execute('not false')).toBe(true);
  });

  it('short-circuits and/or for scalars', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute('false and undefined_var')).toBe(false);
    expect(runtime.execute('true or undefined_var')).toBe(true);
  });

  it('applies logical operators element-wise on tensors', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const andResult = runtime.execute('tensor([1, 0, 1]) and tensor([1, 1, 0])');
    expect(andResult.toArray()).toEqual([1, 0, 0]);
    const orResult = runtime.execute('tensor([1, 0, 0]) or tensor([0, 0, 1])');
    expect(orResult.toArray()).toEqual([1, 0, 1]);
    const notResult = runtime.execute('not tensor([1, 0, 1])');
    expect(notResult.toArray()).toEqual([0, 1, 0]);
  });

  it('evaluates compound assignment on scalars', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute('x = 10\nx += 5')).toBe(15);
    expect(runtime.execute('x = 10\nx -= 3')).toBe(7);
    expect(runtime.execute('x = 10\nx *= 2')).toBe(20);
    expect(runtime.execute('x = 10\nx /= 4')).toBe(2.5);
    expect(runtime.execute('x = 2\nx **= 3')).toBe(8);
  });

  it('evaluates compound assignment on tensors', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute('x = tensor([1, 2, 3])\nx += 1');
    expect(result.toArray()).toEqual([2, 3, 4]);
  });

  it('rejects compound assignment on undefined variable', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(() => runtime.execute('x += 1')).toThrow(/Unknown name 'x'/);
  });

  it('defines and calls user functions', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      fn add(a, b): return a + b
      add(3, 4)
    `)).toBe(7);
  });

  it('supports closures in user functions', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      scale = 10
      fn scaled(x): return x * scale
      scaled(5)
    `)).toBe(50);
  });

  it('supports recursion in user functions', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      fn sum_to(n):
        return n + sum_to(n - 1)
    `)).toBeTypeOf('function');
  });

  it('applies user functions to tensors', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      fn double(x): return x * 2
      double(tensor([1, 2, 3]))
    `);
    expect(result.toArray()).toEqual([2, 4, 6]);
  });

  it('returns last expression when no explicit return', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      fn square(x): x * x
      square(5)
    `)).toBe(25);
  });

  it('evaluates if/elif/else', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute('if true: 1')).toBe(1);
    expect(runtime.execute(`if false: 1
else: 2`)).toBe(2);
    expect(runtime.execute(`if false: 1
elif true: 2
else: 3`)).toBe(2);
    expect(runtime.execute(`if false: 1
elif false: 2
else: 3`)).toBe(3);
  });

  it('evaluates for...in loops', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      total = 0
      for i in [1, 2, 3]: total += i
      total
    `)).toBe(6);
  });

  it('evaluates for...in with range()', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      total = 0
      for i in range(5): total += i
      total
    `)).toBe(10);
  });

  it('evaluates while loops', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      x = 10
      while x > 0: x -= 1
      x
    `)).toBe(0);
  });

  it('supports break in loops', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      total = 0
      for i in range(100):
        if i >= 5: break
        total += i
      total
    `)).toBe(10);
  });

  it('supports continue in loops', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      total = 0
      for i in range(6):
        if i == 3: continue
        total += i
      total
    `)).toBe(12);
  });

  it('propagates return from inside loops', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      fn find_first(items):
        for x in items:
          if x > 3: return x
        return null
      find_first([1, 2, 5, 8])
    `)).toBe(5);
  });

  it('supports nested control flow', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(runtime.execute(`
      total = 0
      for i in range(3):
        for j in range(3):
          if i == j: continue
          total += 1
      total
    `)).toBe(6);
  });

  it('rejects invalid indexing forms', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(() => runtime.execute('tensor([1])[]')).toThrow(/Expected index expression/);
    expect(() => runtime.execute('tensor([1, 2])[::0]')).toThrow(/Slice step must be a positive integer/);
  });
});
