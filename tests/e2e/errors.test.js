import { describe, it, expect } from 'vitest';
import { tensor, matmul, add, Linear, Sequential, compile, CPUTarget } from '../../src/index.js';

const model4x2 = () => new Sequential(new Linear(4, 2));
const x4 = () => tensor([[1, 2, 3, 4]]);

describe('shape errors name the shapes that disagree', () => {
  it('matmul rejects a contracting-dim mismatch instead of returning garbage', () => {
    expect(() => matmul(tensor([[1, 2, 3]]), tensor([[1, 2], [3, 4]])))
      .toThrow(/matmul.*\[1,3\].*\[2,2\]/);
  });

  it('matmul rejects a batched contracting-dim mismatch', () => {
    expect(() => matmul(tensor([[[1, 2, 3]]]), tensor([[[1, 2], [3, 4]]])))
      .toThrow(/matmul/);
  });

  it('Linear rejects an input whose width is not in_features', () => {
    expect(() => new Linear(4, 2).forward(tensor([[1, 2, 3]])))
      .toThrow(/matmul.*\[1,3\].*\[3?4?,/);
  });

  it('elementwise add rejects non-broadcastable shapes', () => {
    expect(() => add(tensor([[1, 2, 3]]), tensor([[1, 2]])))
      .toThrow(/add.*\[1,3\].*\[1,2\]/);
  });

  it('reshape rejects an element-count change', () => {
    expect(() => tensor([[1, 2, 3, 4]]).reshape([3, 3]))
      .toThrow(/reshape.*\[1,4\].*\[3,3\]/);
  });
});

describe('dtype errors list the accepted values', () => {
  it('tensor rejects an unknown dtype', () => {
    expect(() => tensor([1, 2, 3], { dtype: 'not_a_dtype' }))
      .toThrow(/unknown dtype 'not_a_dtype'.*f32/s);
  });

  it('tensor still accepts every advertised dtype', () => {
    for (const dtype of ['f32', 'f64', 'i32', 'bool']) {
      expect(() => tensor([1, 0, 1], { dtype }), `dtype ${dtype} was rejected`).not.toThrow();
    }
  });
});

describe('compiled models reject bad calls with an actionable message', () => {
  it('calling with no arguments says how many inputs are expected', () => {
    const compiled = compile(model4x2(), [x4()], { target: CPUTarget() });
    expect(() => compiled()).toThrow(/expected 1 input tensor\(s\) but got 0/);
  });

  it('calling with too many arguments says how many inputs are expected', () => {
    const compiled = compile(model4x2(), [x4()], { target: CPUTarget() });
    expect(() => compiled(x4(), x4())).toThrow(/expected 1 input tensor\(s\) but got 2/);
  });

  it('passing a non-tensor names the offending argument position', () => {
    const compiled = compile(model4x2(), [x4()], { target: CPUTarget() });
    expect(() => compiled([1, 2, 3, 4])).toThrow(/argument 0 is not a Tensor/);
  });

  it('an input whose width no longer matches the weights is rejected, not silently run', () => {
    const compiled = compile(model4x2(), [x4()], { target: CPUTarget() });
    expect(() => compiled(tensor([[1, 2, 3]]))).toThrow(/mismatch|incompatible/i);
  });

  it('a different batch size still recompiles and runs', async () => {
    const compiled = compile(model4x2(), [x4()], { target: CPUTarget() });
    const out = await compiled(tensor([[1, 2, 3, 4], [5, 6, 7, 8]]));
    expect(out.shape).toEqual([2, 2]);
  });
});
