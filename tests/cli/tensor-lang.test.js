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
      model MLP(input, hidden, output) {
        fc1 = Linear(input, hidden)
        fc2 = Linear(hidden, output)

        forward x {
          x = relu(fc1(x))
          return fc2(x)
        }
      }

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
      model Residual {
        forward x {
          return relu(x + x)
        }
      }
      net = Residual()
      x = randn([2, 4])
      compile(net, input=x)
    `);
    expect(result.result.listKernels().length).toBeGreaterThan(0);
  });

  it('compiles scalar tensor operators inside custom forward', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    const result = runtime.execute(`
      model Scale {
        forward x {
          return mul(x, 2) + 1
        }
      }
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

  it('rejects invalid indexing forms', () => {
    const runtime = new TensorLangRuntime({ output: () => {} });
    expect(() => runtime.execute('tensor([1])[]')).toThrow(/Expected index expression/);
    expect(() => runtime.execute('tensor([1, 2])[::0]')).toThrow(/Slice step must be a positive integer/);
  });
});
