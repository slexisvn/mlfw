import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { teracCompile, teracInvoke, teracRelease } from '#io/terac';
import { TeracRuntimeModule } from '../../../src/backend/terac/runtime_module.js';
import { TeracTarget } from '../../../src/backend/terac/target.js';
import { emitTeraModule } from '../../../src/backend/terac/emit.js';
import { Compiler, compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { parseModule } from '../../../src/compiler/ir/graph/parser.js';
import { verifyModule } from '../../../src/compiler/ir/graph/verifier.js';

vi.mock('#io/terac', () => ({
  teracCompile: vi.fn(), teracInvoke: vi.fn(), teracRelease: vi.fn(),
}));

const modules = [];
const handle = { id: 'native-module' };
const tensor = (data, shape) => ({ data, shape, dtype: 'f32' });

beforeEach(() => {
  vi.resetAllMocks();
  teracCompile.mockReturnValue(handle);
});
afterEach(() => {
  for (const module of modules.splice(0)) module.release();
});

function runtime(options = {}) {
  const module = new TeracRuntimeModule('module {}\n', [
    { name: 'step', inputs: 2, outputs: 1 },
  ], options);
  modules.push(module);
  return module;
}

function graph() {
  const t = new TensorType([2, 3], 'f32');
  const module = new GraphModule('test');
  module.addFunction(buildFunction('add_bias', [t, new TensorType([3], 'f32')], [t], (b, [x, bias]) => {
    b.returnOp(b.add(x, bias).results);
  }));
  return module;
}

describe('TeracRuntimeModule', () => {
  it('passes the original buffer views and current shapes when the same kernel changes width', () => {
    const module = runtime();
    for (const width of [3, 5]) {
      const storage = Float32Array.from({ length: 2 * width + 2 }, (_, i) => i + 1);
      const x = storage.subarray(1, 2 * width + 1);
      const scalar = new Float32Array([2]);
      const outStorage = new Float32Array(2 * width + 2).fill(-99);
      const out = outStorage.subarray(1, 2 * width + 1);
      module.run('step', tensor(x, [2, width]), tensor(scalar, []), tensor(out, [width, 2]));
      const [nativeHandle, name, inputs, inputShape, outputs, outputShape] = teracInvoke.mock.lastCall;
      expect(nativeHandle).toBe(handle);
      expect(name).toBe('step');
      expect(inputs).toHaveLength(2);
      expect(inputs[0]).toBe(x);
      expect(inputs[1]).toBe(scalar);
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toBe(out);
      expect(inputShape).toEqual([2, width]);
      expect(outputShape).toEqual([width, 2]);
    }
    expect(teracCompile).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['missing', [], /no function named missing/],
    ['step', [], /takes 2 tensors and returns 1.*0 buffers/],
    ['step', [tensor({}, [2]), tensor(new Float32Array(1), []), tensor(new Float32Array(2), [2])], /argument 0 is not host memory/],
  ])('rejects invalid call %s before invoking native code', (name, args, message) => {
    expect(() => runtime().run(name, ...args)).toThrow(message);
    expect(teracInvoke).not.toHaveBeenCalled();
  });

  it('releases native memory once and refuses subsequent execution', () => {
    const module = runtime();
    module.release();
    module.release();
    expect(teracRelease).toHaveBeenCalledExactlyOnceWith(handle);
    expect(module.handle).toBeNull();
    expect(() => module.run('step', tensor(new Float32Array([1]), [1]),
      tensor(new Float32Array([2]), []), tensor(new Float32Array(1), [1])))
      .toThrow(/module has been released/);
    expect(teracInvoke).not.toHaveBeenCalled();
  });
});

describe('Terac target and emission', () => {
  it('preserves broadcast results through normalization, printing and parsing', () => {
    const module = graph();
    const mlir = emitTeraModule(module);
    expect(emitTeraModule(module)).toBe(mlir);
    const parsed = parseModule(mlir);
    expect(verifyModule(parsed)).toEqual([]);
    const func = parsed.getFunction('add_bias');
    const add = func.findOp((op) => op.opName === 'add');
    expect(add.operands.map((v) => v.type.shape)).toEqual([[2, 3], [2, 3]]);
    const compiled = compileGraph(func, CPUTarget());
    const output = new Float32Array(6);
    compiled.run('add_bias', new Float32Array([1, -2, 3, 4, -5, 6]), new Float32Array([10, 20, -30]), output);
    expect([...output]).toEqual([11, 18, -27, 14, 15, -24]);
  });

  it('hands normalized graph IR and explicit options to the external compiler', () => {
    const target = TeracTarget({ device: 'cuda', optLevel: 0, library: '/native/tera' });
    const result = new Compiler({ target }).compile(graph());
    modules.push(result.module);
    expect(result.succeeded).toBe(true);
    expect(teracCompile).toHaveBeenCalledTimes(1);
    const [mlir, device, optLevel, options] = teracCompile.mock.lastCall;
    expect(device).toBe('cuda');
    expect(optLevel).toBe(0);
    expect(options.library).toBe('/native/tera');
    const func = parseModule(mlir).getFunction('add_bias');
    expect(func.findOp((op) => op.opName === 'fusion')).toBeNull();
    const output = new Float32Array(6);
    compileGraph(func, CPUTarget()).run('add_bias',
      new Float32Array([2, 4, 6, 8, 10, 12]), new Float32Array([-1, 3, 5]), output);
    expect([...output]).toEqual([1, 7, 11, 7, 13, 17]);
  });
});
